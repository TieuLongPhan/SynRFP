"""Audited72-grid synthesis:48 immutable WL/Path controls plus24 corrected ECFP."""
from __future__ import annotations
import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=Path.home()/'.local/share/synrfp-runs/morgan_ecfp_correction_v1'
EXPORT=HERE/'publication_metadata/morgan_ecfp_correction_v1'
METRICS={'classification':('mcc','accuracy','macro_f1'),
         'clustering':('ami','ari','nmi','silhouette','largest_cluster_fraction','reaction_singleton_fraction','actual_clusters','fit_seconds')}
PRIMARY={'classification':'mcc','clustering':'ami'}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def components():
    # Explicit alias prevents accidentally importing the unrelated sealed22study.
    s=load('morgan43_report_study',HERE/'study.py');previous=sys.modules.get('study');sys.modules['study']=s
    try:
        e=load('morgan43_report_encode',HERE/'encode.py');pe=sys.modules.get('encode');sys.modules['encode']=e
        try:
            m=load('morgan43_report_models',HERE/'models.py');c=load('morgan43_report_clustering',HERE/'clustering.py')
        finally:
            if pe is None:sys.modules.pop('encode',None)
            else:sys.modules['encode']=pe
    finally:
        if previous is None:sys.modules.pop('study',None)
        else:sys.modules['study']=previous
    return s,e,m,c

def analysis():
    folder=HERE.parent/'20_focused_factorial';a=load('morgan43_factorial_analysis',folder/'analysis.py')
    previous=sys.modules.get('analysis');sys.modules['analysis']=a
    try:p=load('morgan43_all_interactions',folder/'per_task_analysis.py')
    finally:
        if previous is None:sys.modules.pop('analysis',None)
        else:sys.modules['analysis']=previous
    return a,p

def source_pins():
    s,_,_,_=components()
    return {str(p.resolve()):s.sha(p) for p in (HERE/'report.py',HERE/'test_report.py',
        HERE.parent/'20_focused_factorial/analysis.py',HERE.parent/'20_focused_factorial/per_task_analysis.py')}

def prepare(root=ROOT,tests=0):
    s,_,_,_=components();s.need(type(tests) is int and tests>0,'Meaningful report tests required')
    value=dict(schema='morgan43-report-code-ready-v1',phase='complete',tests_passed=tests,sources_sha256=source_pins())
    path=Path(root)/'report_CODE_READY.json'
    if path.exists():s.need(s.read(path)==value,'Report readiness changed')
    else:s.write(path,value)
    return value

def keys(task):return ['dataset','config','fold' if task=='classification' else 'seed']

def expected_grid(plan,old,task):
    scope=plan[task+'_datasets'];return set(itertools.product(scope,old['configs'],range(5)))

def validate_frame(frame,plan,old,task,complete=False):
    k=keys(task)
    if not all(c in frame for c in k):raise ValueError('Missing factorial identity')
    actual=list(frame[k].itertuples(index=False,name=None));wanted=expected_grid(plan,old,task)
    if len(set(actual))!=len(actual) or not set(actual).issubset(wanted):raise ValueError('Duplicate/foreign factorial cells')
    if complete and set(actual)!=wanted:raise ValueError('Incomplete72-grid crossing')
    if len(frame) and not np.isfinite(frame[PRIMARY[task]].to_numpy(dtype=float)).all():raise ValueError('Undefined primary metric')
    return frame

def combine(old_frame,new_frame,plan,old,task,complete=True):
    """Never include or relabel old Morgan rows in the corrected crossing."""
    legacy={k for k,v in old['configs'].items() if v['tokenizer'] in ('wl','path')}
    control=old_frame[old_frame.config.isin(legacy)&old_frame.dataset.isin(plan[task+'_datasets'])].copy()
    corrected=new_frame.copy()
    if len(corrected) and not corrected.config.isin(plan['configs']).all():raise ValueError('Fresh results include a non-corrected configuration')
    control['result_origin']='reused-study22-WL-Path';control['tokenizer_profile']='study22-original-wl-path'
    corrected['result_origin']='fresh-study43-ECFP';corrected['tokenizer_profile']=plan['tokenizer_profile']
    frame=pd.concat([control,corrected],ignore_index=True)
    frame['fingerprint']=frame.config
    frame['seed']=frame['fold'].astype(int) if task=='classification' else frame.seed.astype(int)
    for factor in ('tokenizer','radius','sketch','mode'):frame[factor]=[old['configs'][c][factor] for c in frame.config]
    validate_frame(frame,plan,old,task,complete)
    if complete:
        expected=len(plan[task+'_datasets'])*5
        if frame.result_origin.eq('fresh-study43-ECFP').sum()!=24*expected or frame.result_origin.eq('reused-study22-WL-Path').sum()!=48*expected:
            raise ValueError('Corrected/control origin counts differ')
    return frame

def tables(frame,plan,old,task):
    _,p=analysis();metrics=METRICS[task];summary=[]
    for metric in metrics:
        t=frame.groupby(['dataset','config'])[metric].agg(['mean','std','size']).reset_index().rename(columns={'std':'sample_std','size':'repeats'})
        t['metric']=metric;summary.append(t)
    summary=pd.concat(summary,ignore_index=True)
    ranks=summary[summary.metric.eq(PRIMARY[task])].copy();ranks['observed_rank']=ranks.groupby('dataset')['mean'].rank(ascending=False,method='average')
    consistency=ranks.groupby('config').agg(equal_panel_mean=('mean','mean'),mean_observed_rank=('observed_rank','mean'),worst_observed_rank=('observed_rank','max')).reset_index()
    consistency['diagnostic_only_not_config_selection']=True
    effects,refusals=p.full_effects(frame,set(old['configs']),list(metrics))
    groups=['dataset','metric','order','effect','level','reference']
    panels=effects.groupby(groups).contrast.agg(['mean','std','size']).reset_index().rename(columns={'std':'sample_std','size':'paired_repeats'})
    return {'cells.csv':frame,'configuration_summary.csv':summary,'observed_ranks.csv':ranks,
        'cross_panel_consistency.csv':consistency.sort_values('mean_observed_rank'),
        'paired_full_interactions.csv':effects,'panel_full_interactions.csv':panels},refusals

def bind(hashes,cache,path,expected):
    cache.verify(path,expected);hashes[str(Path(path).resolve())]=expected

def native_catalog(s,plan,task,hashes,cache):
    folder=Path(plan['old_wl_path_controls'][task]['path']).parent
    receipt=folder/'COMPLETE.json';complete=s.read(receipt)
    s.need(complete['phase']=='complete' and complete['canonical_metric_and_artifact_checks_passed'] is True
        and complete['plan_sha256']==plan['source_plan_sha256'],'Original native canonical audit is incomplete')
    bind(hashes,cache,receipt,plan['pinned_files'][str(receipt.resolve())])
    path=folder/'inputs.json';bind(hashes,cache,path,complete['outputs_sha256']['inputs.json'])
    return s.read(path)

def stage_catalog(s,root,plan,task,hashes,cache):
    stage='mlp' if task=='classification' else 'clustering';workers=1 if stage=='mlp' else 2;catalog={}
    for worker in range(workers):
        path=root/(stage+'_worker_'+str(worker)+'_COMPLETE.json');r=s.read(path)
        s.need(r['phase']=='complete' and r['stage']==stage and r['worker']==worker and r['workers']==workers
            and r['plan_sha256']==s.sha(root/'plan.json') and r['all_shard_items_verified'] is True
            and r['audited_items']==r['expected_items']==len(plan['tasks'][task])//workers,'Corrected worker shard is not closed')
        s.need(r['sources_sha256']==plan['sources_sha256'],'Worker scientific sources changed')
        for p,h in r['outputs_sha256'].items():
            s.need(p not in catalog,'Overlapping corrected worker shards');catalog[p]=h
        bind(hashes,cache,path,s.sha(path))
    expected={str(s.result_path(root,t)) for t in plan['tasks'][task]}
    s.need(set(catalog)==expected,'Wrong keyed corrected result catalog')
    return catalog

def legacy_closure(s,old,frame,task,hashes,cache,catalog):
    """Revalidate immutable source canonical cells, native rows and artifacts."""
    oldsha=s.sha(s.OLD/'plan.json')
    for row in frame.itertuples():
        t=dict(task=task,dataset=row.dataset,config=row.config,**{('fold' if task=='classification' else 'seed'):int(row.fold if task=='classification' else row.seed)})
        path=s.OLD/'models'/s.task_id(t)/'result.json' if task=='classification' else s.OLD/'clustering'/row.dataset/row.config/(str(row.seed)+'.json')
        r=s.read(path);s.need(r['phase']=='complete' and r.get('identity',r).get('plan_sha256')==oldsha,'Changed original control result')
        for metric in METRICS[task]:
            value=getattr(row,metric);saved=r[metric]
            s.need((pd.isna(value) and saved is None) or np.isclose(value,saved,atol=1e-12,rtol=1e-12,equal_nan=True),'Old canonical metric changed')
        s.need(str(path.resolve()) in catalog,'Original canonical result is not in native closure')
        bind(hashes,cache,path,catalog[str(path.resolve())])
        if task=='classification':
            for name,h in r['artifacts_sha256'].items():bind(hashes,cache,Path(r['attempt_directory'])/name,h)
            with np.load(Path(r['attempt_directory'])/'test_predictions.npz',allow_pickle=False) as z:
                from sklearn.metrics import matthews_corrcoef,accuracy_score,f1_score
                values=dict(mcc=matthews_corrcoef(z['observed'],z['predictions']),accuracy=accuracy_score(z['observed'],z['predictions']),macro_f1=f1_score(z['observed'],z['predictions'],average='macro',zero_division=0))
                s.need(all(np.isclose(values[k],r[k],atol=1e-12,rtol=0) for k in values),'Old prediction metrics do not replay')
        else:
            s.need(r['labels_used_for_fit_or_selection'] is False and r['params']==old['clustering_params'],'Old clustering protocol changed')
            for suffix,key in (('.npz','assignments_sha256'),('.model.npz','centers_sha256')):bind(hashes,cache,path.with_suffix(suffix),r[key])
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                panel=pd.read_csv(old['panels'][row.dataset]['table'],dtype={'row_id':str})
                s.need(np.array_equal(z['row_ids'].astype(str),panel.row_id.to_numpy(dtype=str)) and np.array_equal(z['source_positions'],panel.source_position.to_numpy()),'Old clustering row identities changed')
        src=old['sources'][old['panels'][row.dataset]['source']];receipt=s.read(src['feature_receipt']);fp=Path(src['features'][row.config])
        s.need(r.get('identity',r)['feature_sha256']==receipt['files'][fp.name],'Original result feature binding changed')
        bind(hashes,cache,fp,receipt['files'][fp.name]);bind(hashes,cache,fp.with_suffix('.json'),receipt['files'][fp.with_suffix('.json').name])

def csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()));frame.to_csv(temp,index=False);temp.replace(path)

def report(root=ROOT,task='all'):
    root=Path(root);s,e,m,c=components();cache=s.HashCache();plan=s.verify(root,cache)
    ready=s.read(root/'report_CODE_READY.json');s.need(ready['phase']=='complete' and ready['sources_sha256']==source_pins(),'Changed report source/readiness')
    old=s.read(s.OLD/'plan.json');results={}
    for name in (('classification','clustering') if task=='all' else (task,)):
        rows=[];hashes={};producer=m if name=='classification' else c
        for t in plan['tasks'][name]:
            path=s.result_path(root,t)
            if not path.exists():continue
            r=producer.verify_result(root,plan,t,deep=False,cache=cache)
            row={k:v for k,v in r.items() if not isinstance(v,(dict,list))};row.update(t);row['result_path']=str(path);row['result_sha256']=s.sha(path);rows.append(row)
            bind(hashes,cache,path,s.sha(path))
            if name=='classification':
                for n,h in r['artifacts_sha256'].items():bind(hashes,cache,Path(r['attempt_directory'])/n,h)
            else:
                for suffix,k in (('.npz','assignments_sha256'),('.model.npz','centers_sha256')):bind(hashes,cache,path.with_suffix(suffix),r[k])
        status=dict(schema='morgan43-report-status-v1',phase='running',task=name,corrected_cells=len(rows),expected_corrected_cells=len(plan['tasks'][name]),plan_sha256=s.sha(root/'plan.json'),updated_utc=s.now())
        if len(rows)!=len(plan['tasks'][name]):s.write(root/'report'/name/'status.json',status);results[name]=status;continue
        oldpath=Path(plan['old_wl_path_controls'][name]['path']);cache.verify(oldpath,plan['old_wl_path_controls'][name]['sha256'])
        oldframe=pd.read_csv(oldpath);newframe=pd.DataFrame(rows);frame=combine(oldframe,newframe,plan,old,name)
        legacy=frame[frame.result_origin.eq('reused-study22-WL-Path')]
        original_catalog=native_catalog(s,plan,name,hashes,cache)
        corrected_catalog=stage_catalog(s,root,plan,name,hashes,cache)
        for p,h in corrected_catalog.items():bind(hashes,cache,p,h)
        legacy_closure(s,old,legacy,name,hashes,cache,original_catalog)
        for sid in plan['sources']:
            feature=e.feature_ready(root,plan,sid,cache=cache,deep=True);s.need(feature is not None,'Incomplete corrected feature source')
            fp=Path(plan['sources'][sid]['feature_receipt']);bind(hashes,cache,fp,s.sha(fp))
            for n,h in feature['files'].items():bind(hashes,cache,fp.parent/n,h)
        outputs,refusals=tables(frame,plan,old,name);directory=EXPORT/name
        for n,table in outputs.items():csv(directory/n,table)
        s.write(directory/'metric_refusals.json',refusals)
        protocol=dict(schema='morgan43-corrected72-analysis-v1',task=name,total_configs=72,corrected_morgan_configs=24,reused_wl_path_configs=48,
            tokenizer_profile=s.PROFILE,source_plan_sha256=plan['source_plan_sha256'],plan_sha256=s.sha(root/'plan.json'),core_ready_sha256=plan['core_ready_sha256'],
            row_cap=None,repeats=5,repeat_type='original grouped outer folds' if name=='classification' else 'initialization seeds',
            orders=[1,2,3,4],configuration_selected=None,labels_for_hyperparameter_selection=False,
            uncertainty='Descriptive SD of correlated folds/seeds; no independent-dataset p-values or confidence claims',
            interpretation='Exploratory corrected-tokenizer comparison; old Morgan excluded, exact old WL/Path controls reused')
        s.write(directory/'protocol.json',protocol)
        hashes.update(plan['pinned_files']);hashes.update(plan['sources_sha256']);hashes.update(source_pins());hashes[str(root/'plan.json')]=s.sha(root/'plan.json')
        hashes.update(s.read(root/'CORE_READY.json')['sources_sha256'])
        for p,h in hashes.items():cache.verify(p,h)
        s.write(directory/'inputs.json',hashes)
        text=['# Corrected ECFP construction factorial: '+name,'',f'{len(frame)} audited cells:24fresh ECFP constructions and48immutable WL/Path controls, across{len(plan[name+"_datasets"])}full endpoints and5'+('grouped folds.' if name=='classification' else 'initialization seeds.'),
            '','Old Morgan results are excluded from this versioned72-configuration analysis. The source/core profile and per-row result origins are explicit.',
            'All four interaction orders use the original descriptive reference-contrast mathematics. No configuration is selected automatically; viewed-data exploration is not external confirmation.',
            'Clustering retains raw binary inputs, all original rows and a population-size K rule; this is separate from native SynCluster/common-admitted benchmarking.' if name=='clustering' else 'Original full-row MLP arithmetic, compact selectors and retained weights are preserved; no held-out outcomes enter GPU training.','']
        (directory/'report.md').write_text('\n'.join(text)+'\n')
        names=list(outputs)+['metric_refusals.json','protocol.json','inputs.json','report.md']
        status.update(phase='complete',cells=len(frame),reused_control_cells=len(legacy),all_keyed_cells_verified=True,all_retained_artifact_hashes_verified=True,
            source_and_feature_closure_verified=True,interaction_orders=[1,2,3,4],sources_sha256=source_pins(),outputs_sha256={n:s.sha(directory/n) for n in names})
        s.write(directory/'COMPLETE.json',status)
        local={**status,'outputs_sha256':{**hashes,**{str((directory/n).resolve()):h for n,h in status['outputs_sha256'].items()}}}
        s.write(root/'report'/name/'COMPLETE.json',local);s.write(root/'report'/name/'status.json',status);results[name]=status
    if task=='all' and all(v['phase']=='complete' for v in results.values()):
        receipts={str(root/'report'/n/'COMPLETE.json'):s.sha(root/'report'/n/'COMPLETE.json') for n in results}
        s.write(root/'COMPLETE.json',dict(schema='morgan43-full-correction-complete-v1',phase='complete',plan_sha256=s.sha(root/'plan.json'),
            corrected_mlp_cells=1680,corrected_clustering_cells=1440,combined_classification_cells=5040,combined_clustering_cells=4320,
            tokenizer_profile=s.PROFILE,all_keyed_closures_verified=True,outputs_sha256=receipts,finished_utc=s.now()))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--tests-passed',type=int,default=0)
    p.add_argument('--task',choices=['all','classification','clustering'],default='all');p.add_argument('--once',action='store_true');a=p.parse_args()
    print(json.dumps(prepare(a.root,a.tests_passed) if a.prepare else report(a.root,a.task)))
