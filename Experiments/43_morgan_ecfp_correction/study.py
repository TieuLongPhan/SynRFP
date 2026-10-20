"""Versioned ECFP-only correction of the immutable study22 construction grid."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OLD_HERE=REPO/'Experiments/22_all_endpoint_factorial'
OLD=Path.home()/'.local/share/synrfp-runs/all_endpoint_factorial_full_v1'
ROOT=Path.home()/'.local/share/synrfp-runs/morgan_ecfp_correction_v1'
CORRECTED=REPO.parent/'SynRFP-morgan-ecfp-v004'
PROFILE='synrfp-ecfp-count-v1'
OWN=('study.py','encode.py','models.py','clustering.py','worker.py','test_study.py','test_encode.py','test_models.py','test_clustering.py','PROTOCOL.md')
DATASETS=('schneider_ub','schneider_b','tpl_ub','tpl_b','uspto50k_ub','uspto50k_b','syntemp_r0','syntemp_r1','syntemp_r2','ecreact_1','ecreact_2','ecreact_3','mech_1','mech_2')
CLUSTER_DATASETS=tuple(d for d in DATASETS if d not in ('syntemp_r1','syntemp_r2'))

def need(value,message):
    if not value:raise ValueError(message)

def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def npz(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('wb') as f:np.savez_compressed(f,**values)
    temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def signature(path):
    s=Path(path).stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

class HashCache:
    def __init__(self):self.values={}
    def verify(self,path,expected):
        key=str(Path(path).resolve());before=signature(path)
        if key not in self.values or self.values[key][0]!=before:
            actual=sha(path);need(signature(path)==before,'File changed during hash: '+key)
            self.values[key]=(before,actual)
        need(self.values[key][1]==expected,'Changed pinned file: '+key)
        return expected

def module(name,path,shim=None):
    previous=sys.modules.get('study')
    if shim is not None:sys.modules['study']=shim
    try:
        spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
        sys.modules[name]=value;spec.loader.exec_module(value);return value
    finally:
        if shim is not None:
            if previous is None:sys.modules.pop('study',None)
            else:sys.modules['study']=previous

def legacy_shim():
    # Legacy GPU paths stay legacy; every plan/storage/hash operation is new.
    return SimpleNamespace(HERE=OLD_HERE,ROOT=ROOT,sha=sha,read=read,write=write,npz=npz,now=now,verify=verify)

def sources():
    paths=[HERE/n for n in OWN]+[OLD_HERE/n for n in ('models.py','clustering.py','gpu_worker.py','gpu_worker_runtime_v2.py')]
    paths += [REPO/'Experiments/20_focused_factorial/encode_factorial.py',REPO/'Experiments/16_modes/classification_worker.py']
    return {str(p.resolve()):sha(p) for p in paths}

def core_ready(root=ROOT,cache=None):
    cache=cache or HashCache();path=Path(root)/'CORE_READY.json';r=read(path)
    need(r.get('phase')=='complete','Corrected core not ready')
    need(r.get('tokenizer_profile',r.get('profile'))==PROFILE,'Wrong ECFP profile')
    directory=Path(r.get('corrected_root',r.get('worktree',str(CORRECTED)))).resolve()
    need(directory==CORRECTED.resolve(),'Wrong corrected core worktree')
    pins=r.get('sources_sha256',{})
    need(pins and str((directory/'synrfp/tokenizers/morgan.py').resolve()) in pins,'Missing corrected Morgan pin')
    for p,h in pins.items():cache.verify(p,h)
    return r

def task_id(t):
    repeat='f'+str(t['fold']) if t['task']=='classification' else 's'+str(t['seed'])
    return t['task']+'__'+t['dataset']+'__'+repeat+'__'+t['config']

def result_path(root,t):
    if t['task']=='classification':return Path(root)/'models'/task_id(t)/'result.json'
    return Path(root)/'artifacts/clustering'/t['dataset']/t['config']/(str(t['seed'])+'.json')

def validate_grid(plan):
    need(plan['schema']=='synrfp-morgan-ecfp-correction-v1' and plan['tokenizer_profile']==PROFILE,'Wrong correction schema/profile')
    need(len(plan['classification_datasets'])==14 and set(plan['classification_datasets'])==set(DATASETS)
        and len(plan['clustering_datasets'])==12 and set(plan['clustering_datasets'])==set(CLUSTER_DATASETS),'Wrong endpoint scope')
    levels={(c['tokenizer'],c['radius'],c['sketch'],c['mode']) for c in plan['configs'].values()}
    need(levels==set(itertools.product(('morgan',),(2,3),('parity','minhash','cw'),('delta','union','delta_union','roles'))),'Wrong24-configuration grid')
    for name,repeat,expected in [('classification','fold',1680),('clustering','seed',1440)]:
        tasks=plan['tasks'][name];scope=plan[name+'_datasets']
        actual={(t['dataset'],t['config'],t[repeat]) for t in tasks}
        need(len(tasks)==expected and actual==set(itertools.product(scope,plan['configs'],range(5))),'Changed keyed '+name+' grid')
        need(all(t['task']==name for t in tasks),'Mixed task identity')
    need(len(plan['sources'])==9 and plan['row_cap'] is None and plan['new_filtering'] is False,'Changed source population')
    return plan

def prepare(root=ROOT):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():return verify(root)
    core_ready(root);old=read(OLD/'plan.json');oldsha=sha(OLD/'plan.json')
    need(read(OLD/'plan_seal.json')['sha256']==oldsha,'Changed source plan seal')
    configs={k:v for k,v in old['configs'].items() if v['tokenizer']=='morgan'}
    panels={d:old['panels'][d] for d in DATASETS};used={p['source'] for p in panels.values()}
    pins={str((OLD/'plan.json').resolve()):oldsha,str((root/'CORE_READY.json').resolve()):sha(root/'CORE_READY.json')}
    src={}
    for name in sorted(used,key=lambda s:(old['sources'][s]['n'],s)):
        original=old['sources'][name];anchor=Path(original['features']['synrfp-wl-r2-b2048-parity-delta'])
        receipt=read(original['feature_receipt']);need(receipt['phase']=='complete' and receipt['plan_sha256']==oldsha,'Original validity source incomplete')
        pins[original['table']]=old['pinned_files'][original['table']]
        pins[str(anchor)]=receipt['files'][anchor.name]
        pins[original['feature_receipt']]=sha(original['feature_receipt'])
        src[name]=dict(table=original['table'],n=original['n'],parser_profile='classification',validity_source=str(anchor),
            validity_source_sha256=receipt['files'][anchor.name],features={c:str(root/'artifacts/features'/name/(c+'.npz')) for c in configs},
            feature_receipt=str(root/'artifacts/features'/name/'COMPLETE.json'))
    for p in panels.values():
        for path in [p['table'],*p['split_paths'].values()]:pins[path]=old['pinned_files'][path]
    bindings={}
    for task in ('classification','clustering'):
        folder=OLD_HERE/'publication_metadata/all_endpoint_factorial_full_v1'/task
        r=read(folder/'COMPLETE.json');need(r['phase']=='complete','Original factorial incomplete')
        for name in ('COMPLETE.json','cells.csv'):
            path=folder/name;pins[str(path.resolve())]=sha(path)
        need(sha(folder/'cells.csv')==r['outputs_sha256']['cells.csv'],'Changed original canonical cells')
        bindings[task]=dict(path=str(folder/'cells.csv'),sha256=sha(folder/'cells.csv'),reused_tokenizers=['wl','path'],reused_configs=48)
    storage=read(root/'ARTIFACT_STORAGE.json');need(storage['phase']=='complete' and storage['existing_files_moved_or_deleted'] is False,'Storage not prepared')
    need((root/'artifacts').resolve()==Path(storage['physical_artifacts_path']).resolve(),'Artifact storage mismatch')
    pins[str(root/'ARTIFACT_STORAGE.json')]=sha(root/'ARTIFACT_STORAGE.json')
    tasks={k:[t for t in old['tasks'][k] if t['dataset'] in panels and t['config'] in configs] for k in ('classification','clustering')}
    plan=dict(schema='synrfp-morgan-ecfp-correction-v1',created_utc=now(),root=str(root),tokenizer_profile=PROFILE,
        corrected_root=str(CORRECTED),core_ready_sha256=sha(root/'CORE_READY.json'),source_plan_sha256=oldsha,
        configs=configs,panels=panels,sources=src,classification_datasets=list(DATASETS),clustering_datasets=old['clustering_datasets'],tasks=tasks,
        counts={k:len(v) for k,v in tasks.items()},feature_pools=216,row_cap=None,new_filtering=False,
        params=old['params'],clustering_params=old['clustering_params'],clustering_k=old['clustering_k'],
        failure_policy='Freeze original all-tokenizer common validity; retain old invalid zero rows; any corrected construction failure on an old-valid row is fatal',
        parser_profile='classification; agents both',stereo_features=False,seed=1,bits=2048,
        old_wl_path_controls=bindings,sources_sha256=sources(),pinned_files=pins,
        interpretation='Versioned Morgan correction; old WL/Path controls reused, old Morgan historical; exploratory scores, no new external confirmation')
    validate_grid(plan);write(root/'plan.json',plan);write(root/'plan_seal.json',{'sha256':sha(root/'plan.json')});return verify(root)

def verify(root=ROOT,cache=None):
    root=Path(root);cache=cache or HashCache();p=read(root/'plan.json');validate_grid(p)
    cache.verify(root/'plan.json',read(root/'plan_seal.json')['sha256'])
    for path,h in {**p['pinned_files'],**p['sources_sha256']}.items():cache.verify(path,h)
    core_ready(root,cache);return p

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
    result=prepare(a.root) if a.prepare else verify(a.root);print(json.dumps({'phase':'complete','counts':result['counts'],'plan_sha256':sha(a.root/'plan.json')}))
