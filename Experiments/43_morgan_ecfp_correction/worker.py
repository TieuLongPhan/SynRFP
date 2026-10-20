"""Cooperative, resumable corrected-Morgan producers; no resource-policy mutation."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
import study as S
import encode as E
import models as M
import clustering as C

YIELD=False

def request_yield(signum,frame):
    global YIELD
    YIELD=True

def review(root,plan,stage):
    r=S.read(Path(root)/'ROOT_REVIEW.json')
    S.need(r['schema']=='morgan43-root-reviewed-v1' and r['phase']=='reviewed' and r['plan_sha256']==S.sha(Path(root)/'plan.json'),'Missing reviewed execution')
    S.need(stage in r['stages_allowed'],'Stage not reviewed')
    for p,h in plan['sources_sha256'].items():S.need(r['sources_sha256'].get(p)==h,'Review omits scientific source')
    for p,h in r['sources_sha256'].items():S.need(S.sha(p)==h,'Reviewed execution source changed')
    return r

def stage_items(plan,stage,worker,workers,source=None):
    if stage=='encode':
        values=sorted(plan['sources'],key=lambda x:(plan['sources'][x]['n'],x))
        if source is not None:
            S.need(source in values,'Unknown source');values=[source]
    else:
        name='classification' if stage=='mlp' else 'clustering'
        values=sorted(plan['tasks'][name],key=lambda t:(plan['panels'][t['dataset']]['n'],t['config'],S.task_id(t)))
    return [v for i,v in enumerate(values) if i%workers==worker]

def verify_item(root,plan,stage,item,deep=False):
    if stage=='encode':return E.feature_ready(root,plan,item,cache=CACHE,deep=deep)
    return (M if stage=='mlp' else C).verify_result(root,plan,item,deep=deep,cache=CACHE)

def item_output(root,plan,stage,item):
    return Path(plan['sources'][item]['feature_receipt']) if stage=='encode' else S.result_path(root,item)

CACHE=S.HashCache()

def run(root=S.ROOT,stage='encode',worker=0,workers=2,threads=2,run_id=None,source=None,encode_processes=None):
    root=Path(root);plan=S.verify(root,cache=CACHE);review(root,plan,stage)
    S.need(0<=worker<workers and threads>=1,'Invalid worker profile')
    run_id=run_id or stage+'-w'+str(worker)+'-'+str(os.getpid())
    S.need(Path(run_id).name==run_id,'Invalid invocation ID')
    invocation=root/'workers'/run_id
    S.need(not invocation.exists(),'Invocation ID already retained')
    invocation.mkdir(parents=True)
    signal.signal(signal.SIGUSR1,request_yield)
    base=dict(stage=stage,worker=worker,workers=workers,run_id=run_id,plan_sha256=S.sha(root/'plan.json'),sources_sha256=plan['sources_sha256'])
    S.write(invocation/'STARTED.json',{**base,'phase':'ready','pid':os.getpid(),'started_utc':S.now()})
    lease=None
    if stage=='mlp':
        lease_path=S.REPO/'Experiments/12_validation/runs/gpu_1.lock'
        S.need(lease_path.is_file(),'Physical GPU1 lease missing; do not create/recreate')
        lease=lease_path.open('r+')
        while True:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if YIELD:break
                time.sleep(5)
    items=stage_items(plan,stage,worker,workers,source);verified={}
    try:
        while not YIELD and not (root/'HOLD_NEW_FITS').exists():
            pending=False;launched=False
            for item in items:
                path=item_output(root,plan,stage,item)
                if str(path) in verified:continue
                if verify_item(root,plan,stage,item) is not None:
                    verified[str(path)]=S.sha(path);continue
                pending=True
                if stage!='encode':
                    sid=plan['panels'][item['dataset']]['source']
                    if E.feature_ready(root,plan,sid,cache=CACHE) is None:continue
                key=('encode__'+item) if stage=='encode' else S.task_id(item)
                claim=root/'claims'/(key+'.lock');claim.parent.mkdir(parents=True,exist_ok=True)
                with claim.open('a+') as f:
                    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if verify_item(root,plan,stage,item) is None:
                        S.write(root/(stage+'_worker_'+str(worker)+'_status.json'),{**base,'phase':'working','item':item,'updated_utc':S.now()})
                        if stage=='encode':E.encode_source(root,plan,item,encode_processes or threads)
                        elif stage=='mlp':M.fit(root,plan,item,1)
                        else:C.fit(root,plan,item,threads)
                    S.need(verify_item(root,plan,stage,item,deep=True) is not None,'Unclosed scientific result')
                    verified[str(path)]=S.sha(path);launched=True
                break
            if not pending:break
            if not launched:time.sleep(10)
        all_done=len(verified)==len(items)
        receipt={**base,'phase':'complete' if all_done else 'yielded','expected_items':len(items),'audited_items':len(verified),
            'all_shard_items_verified':all_done,'outputs_sha256':verified,'finished_utc':S.now()}
        S.write(invocation/('COMPLETE.json' if all_done else 'YIELDED.json'),receipt)
        if all_done and source is None:S.write(root/(stage+'_worker_'+str(worker)+'_COMPLETE.json'),receipt)
        return receipt
    finally:
        if lease is not None:lease.close()

def pilot(root=S.ROOT):
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    import resource
    root=Path(root);plan=S.verify(root);f=pd.read_csv(plan['sources']['mech']['table'],dtype={'row_id':str}).iloc[:64]
    with np.load(plan['sources']['mech']['validity_source'],allow_pickle=False) as z:valid=z['ok'][:len(f)]
    arrays=E.encode_reactions(f.rxn.astype(str).tolist(),valid,root)
    directory=root/'pilot';directory.mkdir(parents=True,exist_ok=True)
    S.npz(directory/'features.npz',**arrays,ok=valid,source_positions=f.source_position.to_numpy(),row_ids=f.row_id.to_numpy(dtype=str))
    config=next(iter(plan['configs']));X=np.unpackbits(arrays[config],axis=1,count=2048).astype(np.float32)
    k=min(512,max(2,int(round(np.sqrt(len(f)/2)))))
    with threadpool_limits(limits=2):
        m=MiniBatchKMeans(n_clusters=k,random_state=0,**plan['clustering_params']);labels=m.fit_predict(X)
        S.need(np.array_equal(labels,m.predict(X)),'Pilot center replay failed')
    S.npz(directory/'clustering.npz',labels=labels,centers=m.cluster_centers_)
    classes,state=M.M.initialization(np.tile([0,1],32),plan['params']['classification'],0)
    S.npz(directory/'initialization.npz',classes=classes,**state)
    r=dict(schema='morgan43-cpu-pilot-v1',phase='complete',passed=True,plan_sha256=S.sha(root/'plan.json'),sources_sha256=plan['sources_sha256'],
        tokenizer_profile=S.PROFILE,actual_encoding=True,encoded_rows=len(f),configurations=24,actual_clustering=True,
        center_replay_passed=True,original_mlp_initialization_executed=True,gpu_training_executed=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        outputs_sha256={str(p):S.sha(p) for p in directory.glob('*.npz')},finished_utc=S.now())
    S.write(root/'PILOT_READY.json',r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=S.ROOT);p.add_argument('--stage',choices=['encode','mlp','clustering'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--run-id');p.add_argument('--source');p.add_argument('--encode-processes',type=int);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.pilot:pilot(a.root)
    else:
        S.need(a.stage is not None,'Choose stage or pilot');run(a.root,a.stage,a.worker,a.workers,a.threads,a.run_id,a.source,a.encode_processes)
