"""Full-row72-configuration construction study, separate from bounded pilots."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path.home() / '.local/share/synrfp-runs/all_endpoint_factorial_full_v1'
EXPORT = HERE / 'publication_metadata/all_endpoint_factorial_full_v1'
FOCUSED = HERE.parent / '20_focused_factorial'
for p in [FOCUSED, HERE.parent / '12_validation', HERE.parent / '13_yield', REPO]:
    sys.path.insert(0, str(p))
import encode_factorial as E


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **values)
    tmp.replace(path)


def pin(pins, path):
    path = Path(path).resolve(strict=True); pins[str(path)] = sha(path); return str(path)


def table(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={'method':'gzip','mtime':0})


def verify(root=ROOT, inputs=False):
    root = Path(root)
    p = json.loads((root / 'plan.json').read_text())
    if p['schema'] != 'synrfp-all-endpoint-full-factorial-v1' or sha(root / 'plan.json') != json.loads((root / 'plan_seal.json').read_text())['sha256']:
        raise ValueError('Changed full-row factorial plan')
    for path, digest in p['pinned_files'].items():
        if sha(path) != digest: raise ValueError('Changed source/population: ' + path)
    if inputs:
        for path, digest in p['input_files'].items():
            if sha(path) != digest: raise ValueError('Changed original input: ' + path)
    return p


def panel(root, name, source, frame, codes, pins, family):
    path = root / 'panels' / (name + '.csv.gz'); table(path, frame); pin(pins,path)
    splits = {}
    for fold, values in codes.items():
        q = root / 'splits' / name / (str(fold) + '.npy'); q.parent.mkdir(parents=True,exist_ok=True)
        np.save(q,values,allow_pickle=False); splits[str(fold)] = pin(pins,q)
    return {'dataset':name,'source':source,'table':str(path),'n':len(frame),
            'split_paths':splits,'family':family,'row_budget':'full original endpoint population'}


def prepare(root=ROOT):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    if (root / 'plan.json').exists(): return verify(root,inputs=True)
    import run_crxnfp_full as B
    import run_continuous as Y
    parent = B.verify(); cfg = B.frozen_cfg(); yp = Y.verify()
    pins, inputs, sources, panels = {}, {}, {}, {}
    for p in HERE.glob('*.py'): pin(pins,p)
    for p in [FOCUSED/'encode_factorial.py',HERE.parent/'16_modes/classification_worker.py',
              HERE.parent/'03_classification/mlp_gpu_queue.py',HERE.parent/'13_yield/run_continuous.py',
              HERE.parent/'13_yield/yield_retrain_cpu.py']:
        pin(pins,p)
    for p in (REPO/'synrfp').rglob('*.py'): pin(pins,p)
    for p in [B.ROOT/'plan.json',B.ROOT/'config.yaml',Y.ROOT/'plan.json']: pin(inputs,p)
    # Small yield sources first so GPU timing can start during larger encodings.
    yield_splits = []
    for d,spec in yp['tables'].items():
        pin(inputs,spec['path']); full = pd.read_csv(spec['path'],dtype={'row_id':str})
        coverage = Y.ROOT / ('coverage_' + d + '.npz'); pin(inputs,coverage)
        with np.load(coverage,allow_pickle=False) as z: rows = np.flatnonzero(z['shared'])
        f = full.iloc[rows].reset_index(drop=True); f['source_position'] = rows
        f['row_id'] = f.row_id.astype(str); f['rxn'] = f.left.astype(str) + '>>' + f.right.astype(str)
        path = root/'sources'/(d+'.csv.gz'); table(path,f);pin(pins,path)
        sources[d] = {'table':str(path),'n':len(f),'parser_profile':'yield','anchors':{},
                      'features':{fp:str(root/'features'/d/(fp+'.npz')) for fp in E.VARIANTS},
                      'feature_receipt':str(root/'features'/d/'COMPLETE.json'),
                      'population':'Full original seven-FP shared-valid population; no cap or new exclusions',
                      'original_raw_rows':len(full)}
        panels[d] = {'dataset':d,'source':d,'table':str(path),'n':len(f),'family':'yield'}
        for old in [s for s in yp['splits'] if s['dataset']==d]:
            pin(inputs,old['path'])
            with np.load(old['path'],allow_pickle=False) as z:
                parts = {part:np.flatnonzero(np.isin(rows,z[part])).astype(np.int64) for part in ['train','validation','test']}
            if not np.array_equal(np.sort(np.concatenate(list(parts.values()))),np.arange(len(f))):
                raise ValueError('Yield partitions do not cover full original population')
            q=root/'splits'/d/(old['split']+'.npz');npz(q,**parts);pin(pins,q)
            yield_splits.append({'dataset':d,'split':old['split'],'path':str(q),'report_group':old['report_group']})
        anchors = {(2,'delta'):('r2_2048_check_v1','synrfp_r2b2048'),
                   (3,'delta'):('retrain_r3_union_v1','synrfp_r3b2048'),
                   (2,'union'):('roles_v1','synrfp_union_r2b2048'),
                   (3,'union'):('retrain_r3_union_v1','synrfp_union_r3b2048'),
                   (2,'delta_union'):('roles_v1','synrfp_du_r2b2048'),
                   (3,'delta_union'):('roles_v1','synrfp_du_r3b2048'),
                   (2,'roles'):('roles_v1','synrfp_roles_r2b2048'),
                   (3,'roles'):('roles_v1','synrfp_roles_r3b2048')}
        for (radius,mode),(run,fp) in anchors.items():
            p=HERE.parent/'13_yield/runs'/run/'features'/d/(fp+'.npz')
            sources[d]['anchors'][E.variant_name('wl',radius,'parity',mode)]={'path':pin(inputs,p),'positions':rows.tolist()}
    class_datasets = parent['classification_datasets']
    for d in class_datasets:
        ds = B.dataset_table(cfg,d); key = B.rxn_key(cfg,d)
        pin(inputs,cfg['paths']['work']/'tables'/('ds_'+d+'.pkl'))
        if key not in sources:
            rt = B.reaction_table(cfg,key); pin(inputs,cfg['paths']['work']/'tables'/('rxn_'+key+'.pkl'))
            f = pd.DataFrame({'row_id':rt.r_id.astype(str),'source_position':np.arange(len(rt),dtype=np.int64),'rxn':rt.rxn.astype(str)})
            path=root/'sources'/(key+'.csv.gz');table(path,f);pin(pins,path)
            wide=HERE.parent/'12_validation/runs/classification_r2_2048_v1/features'/key/'synrfp-wl-r2-b2048-parity-delta.npz'
            sources[key]={'table':str(path),'n':len(f),'parser_profile':'classification',
                          'features':{fp:str(root/'features'/key/(fp+'.npz')) for fp in E.VARIANTS},
                          'feature_receipt':str(root/'features'/key/'COMPLETE.json'),
                          'anchors':{E.variant_name('wl',2,'parity','delta'):{'path':pin(inputs,wide),'positions':list(range(len(f)))}}}
        source = pd.read_csv(sources[key]['table'],dtype={'row_id':str})
        rows=ds.row.to_numpy(dtype=np.int64)
        if not np.array_equal(ds.r_id.astype(str).to_numpy(),source.row_id.to_numpy()[rows]):
            raise ValueError('Full classification source identity mismatch')
        f=pd.DataFrame({'row_id':ds.r_id.astype(str),'source_position':rows,'dataset_position':np.arange(len(ds)),
                        'y':ds.y.to_numpy(dtype=np.int64)})
        codes={}
        for fold in range(5):
            old=cfg['paths']['work']/'splits'/d/f'cv_s{fold}.npy';pin(inputs,old)
            codes[fold]=B.code(cfg,d,fold)
        panels[d]=panel(root,d,key,f,codes,pins,'classification')
    # EHReact is a separate binary case study using every row of each eligible assay.
    raw_path=HERE.parent/'10_enzyme_activity/results/ehreact_raw.csv.gz';pin(inputs,raw_path)
    archive_path=HERE.parent/'10_enzyme_activity/results/grouped_folds.npz';pin(inputs,archive_path)
    raw=pd.read_csv(raw_path);unique=pd.Index(raw.rxn.astype(str).unique());source_key='ehreact_activity_full'
    f=pd.DataFrame({'row_id':[hashlib.sha256(x.encode()).hexdigest() for x in unique],
                    'source_position':np.arange(len(unique),dtype=np.int64),'rxn':unique})
    path=root/'sources'/(source_key+'.csv.gz');table(path,f);pin(pins,path)
    sources[source_key]={'table':str(path),'n':len(f),'parser_profile':'classification','anchors':{},
                        'features':{fp:str(root/'features'/source_key/(fp+'.npz')) for fp in E.VARIANTS},
                        'feature_receipt':str(root/'features'/source_key/'COMPLETE.json'),
                        'population':'Original EHReact raw reaction strings, identical to archived activity inputs'}
    activity=[]
    with np.load(archive_path,allow_pickle=False) as z:
        for (system,assay),sub in raw.groupby(['system','assay'],sort=True):
            y=sub.activity.to_numpy(dtype=np.int64)
            if y.sum()<10 or len(y)-y.sum()<10: continue
            name='ehreact__'+system+'__'+assay;activity.append(name)
            rows=unique.get_indexer(sub.rxn.astype(str))
            f=pd.DataFrame({'row_id':[f'{system}|{assay}|{i}|{rid}' for i,rid in zip(sub.index,sub.r_id)],
                            'source_position':rows,'dataset_position':np.arange(len(sub)),'y':y})
            codes={}
            for fold in range(5):
                prefix=f'{system}__{assay}__r0__f{fold}'
                tr,te=z[prefix+'__tr'],z[prefix+'__te']
                if not np.array_equal(np.sort(np.r_[tr,te]),np.arange(len(sub))):raise ValueError('Changed EHReact fold coverage')
                code=np.zeros(len(sub),dtype=np.int8);code[te]=2;codes[fold]=code
            panels[name]=panel(root,name,source_key,f,codes,pins,'activity')
    config=dict(E.VARIANTS)
    tasks={'classification':[{'task':'classification','dataset':d,'config':fp,'fold':fold}
            for d in class_datasets+activity for fp in config for fold in range(5)],
           'yield':[{'task':'yield','dataset':s['dataset'],'config':fp,'split':s['split']}
                    for s in yield_splits for fp in config],
           'clustering':[{'task':'clustering','dataset':d,'config':fp,'seed':seed}
                         for d in parent['clustering_datasets'] for fp in config for seed in range(5)]}
    plan={'schema':'synrfp-all-endpoint-full-factorial-v1','created_utc':now(),'root':str(root),
          'pinned_files':pins,'input_files':inputs,'configs':config,'sources':sources,'panels':panels,
          'classification_datasets':class_datasets,'activity_datasets':activity,
          'clustering_datasets':parent['clustering_datasets'],'yield_splits':yield_splits,'tasks':tasks,
          'counts':{k:len(v) for k,v in tasks.items()},'params':{'classification':cfg['models']['mlp'],'yield':Y.PARAMS},
          'clustering_params':{'batch_size':8192,'n_init':3,'max_iter':100,'reassignment_ratio':.01,'tol':0.,'max_no_improvement':10},
          'clustering_k':'min(512,max(2,round(sqrt(n/2)))) based on full endpoint rowcount only',
          'encoder_protocol':E.PROTOCOL,'row_cap':None,'fingerprint_seed':1,'baseline_fits':0,
          'repeat_scope':'One original five-fold grouped CV for classification; five clustering seeds; all64 original yield partitions; EHReact first original five-fold repeat per assay',
          'failure_policy':'Classification and clustering retain all original rows including zero-vector encoding refusals; yield retains exact original shared-valid population',
          'backend':'Fresh matched FP32 GPU training for classification/yield; results kept separate from pilot CPU scores',
          'storage':'Shared packed feature pools and compact per-model row/state/weight/prediction records; no copied dense matrices per fit',
          'retention':'All original/model weights retained locally; no model publishing',
          'interpretation':'Exploratory factorial after prior scores viewed; no automatic test-score selection or independent-fold significance claims',
          'barrier_stereo':False}
    write(root/'plan.json',plan);write(root/'plan_seal.json',{'sha256':sha(root/'plan.json')});write(EXPORT/'plan.json',plan)
    verify(root,inputs=True);return plan


def chunk(arg):
    strings,profile=arg; return E.encode_reactions(strings,profile)


def encode(root=ROOT,workers=4):
    root=Path(root);plan=verify(root,inputs=True)
    for name,s in plan['sources'].items():
        receipt=Path(s['feature_receipt'])
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['plan_sha256']!=sha(root/'plan.json'):raise ValueError('Changed feature plan')
            for n,h in old['files'].items():
                if sha(receipt.parent/n)!=h:raise ValueError('Changed completed full feature pool')
            continue
        if shutil.disk_usage(root).free<40*1024**3:raise RuntimeError('Below40GiB free-space reserve')
        f=pd.read_csv(s['table'],dtype={'row_id':str}); strings=f.rxn.astype(str).tolist(); receipt.parent.mkdir(parents=True,exist_ok=True)
        matrices={fp:np.lib.format.open_memmap(receipt.parent/(fp+'.partial.npy'),mode='w+',dtype=np.uint8,shape=(len(f),256)) for fp in plan['configs']}
        ok=np.zeros(len(f),bool)
        pieces=[(strings[i:i+128],s['parser_profile']) for i in range(0,len(f),128)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i,(encoded,valid) in enumerate(pool.map(chunk,pieces)):
                a=i*128;b=a+len(valid);ok[a:b]=valid
                for fp,X in encoded.items():matrices[fp][a:b]=X
                if i%20==0:write(root/'feature_status.json',{'phase':'encoding','source':name,'rows_encoded':b,'expected':len(f),'workers':workers,'updated_utc':now()})
        for fp,anchor in s['anchors'].items():
            with np.load(anchor['path'],allow_pickle=False) as z:
                take=np.asarray(anchor['positions'],dtype=np.int64)
                if not np.array_equal(matrices[fp],z['X'][take]) or not np.array_equal(ok,z['ok'][take]):
                    raise ValueError('Full-row native-anchor mismatch: '+name+'/'+fp)
        files={}
        for fp,X in matrices.items():
            X.flush();path=Path(s['features'][fp])
            npz(path,X=X,ok=ok,n_bits=np.asarray(2048),source_positions=f.source_position.to_numpy(dtype=np.int64),row_ids=f.row_id.to_numpy(dtype=str))
            metadata={'fingerprint':fp,'spec':plan['configs'][fp],'plan_sha256':sha(root/'plan.json'),
                      'source_table_sha256':sha(s['table']),'rowcount':len(f),'feature_sha256':sha(path),
                      'parser_profile':s['parser_profile'],'native_anchor_checks':list(s['anchors']),
                      'encoder_source_sha256':sha(FOCUSED/'encode_factorial.py'),'failed_rows':int((~ok).sum())}
            write(path.with_suffix('.json'),metadata)
            files[path.name]=sha(path);files[path.with_suffix('.json').name]=sha(path.with_suffix('.json'))
            # Disposable encoding scratch; model weights and published caches are preserved.
            scratch=receipt.parent/(fp+'.partial.npy');del X;matrices[fp]=None;scratch.unlink()
        write(receipt,{'phase':'complete','plan_sha256':sha(root/'plan.json'),'files':files,'rows':len(f),'failed_rows':int((~ok).sum()),'finished_utc':now()})
    write(root/'FEATURES_COMPLETE.json',{'phase':'complete','sources':len(plan['sources']),'plan_sha256':sha(root/'plan.json'),'finished_utc':now()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','verify','features'],required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    if a.stage=='prepare': print(json.dumps(prepare(a.root)['counts']))
    elif a.stage=='features':encode(a.root,a.workers)
    else:verify(a.root,inputs=True);print('Full-row plan verified')
