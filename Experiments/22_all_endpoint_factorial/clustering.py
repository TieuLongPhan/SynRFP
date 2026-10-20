"""Full-row label-blind construction factorial with retained centers/labels."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,normalized_mutual_info_score,silhouette_score
from threadpoolctl import threadpool_limits

import study as S


def key(t):return t['dataset']+'__'+t['config']+'__s'+str(t['seed'])


def output(root,t):return Path(root)/'clustering'/t['dataset']/t['config']/(str(t['seed'])+'.json')


def data(plan,t):
    panel=plan['panels'][t['dataset']];s=plan['sources'][panel['source']]
    receipt=Path(s['feature_receipt'])
    if not receipt.exists():return None
    fp=Path(s['features'][t['config']]);h=json.loads(receipt.read_text())
    if h['files'][fp.name]!=S.sha(fp):raise ValueError('Changed full clustering feature pool')
    f=pd.read_csv(panel['table'],dtype={'row_id':str});rows=f.source_position.to_numpy(dtype=np.int64)
    with np.load(fp,allow_pickle=False) as z:
        X=np.unpackbits(z['X'][rows],axis=1,count=2048).astype(np.float32);ok=z['ok'][rows]
    if X.shape!=(len(f),2048) or not np.isfinite(X).all():raise ValueError('Malformed full clustering matrix')
    return f,X,ok,fp


def external_scores(y,labels):
    return {'ami':float(adjusted_mutual_info_score(y,labels)),'ari':float(adjusted_rand_score(y,labels)),
            'nmi':float(normalized_mutual_info_score(y,labels))}


def verify_result(root,plan,t,record,replay=False):
    path=output(root,t)
    if record['plan_sha256']!=S.sha(Path(root)/'plan.json') or record['source_sha256']!=S.sha(__file__):raise ValueError('Changed clustering identity')
    for suffix,field in [('.npz','assignments_sha256'),('.model.npz','centers_sha256')]:
        if S.sha(path.with_suffix(suffix))!=record[field]:raise ValueError('Changed clustering artifact')
    if replay:
        f,X,ok,fp=data(plan,t)
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:labels=z['labels']
        with np.load(path.with_suffix('.model.npz'),allow_pickle=False) as z:centers=z['centers']
        model=MiniBatchKMeans(n_clusters=len(centers));model.cluster_centers_=centers;model._n_threads=1;model.n_features_in_=2048
        with threadpool_limits(limits=1):actual=model.predict(X)
        if not np.array_equal(labels,actual):raise ValueError('Full clustering center replay mismatch')
        for k,v in external_scores(f.y.to_numpy(),labels).items():
            if not np.isclose(record[k],v,atol=1e-12,rtol=0):raise ValueError('Full clustering metric replay mismatch')


def fit(root,plan,t,threads=4):
    path=output(root,t)
    if path.exists():verify_result(root,plan,t,json.loads(path.read_text()));return True
    ready=data(plan,t)
    if ready is None:return False
    f,X,ok,fp=ready;n=len(f);k=min(512,max(2,int(round(np.sqrt(n/2)))))
    params=plan['clustering_params'];start=time.monotonic()
    with threadpool_limits(limits=threads):
        model=MiniBatchKMeans(n_clusters=k,random_state=int(t['seed']),**params)
        labels=model.fit_predict(X)
    fit_seconds=time.monotonic()-start
    # Original source order and every failed-encoding row are retained.
    path.parent.mkdir(parents=True,exist_ok=True)
    S.npz(path.with_suffix('.npz'),labels=labels.astype(np.int32),row_ids=f.row_id.to_numpy(dtype=str),source_positions=f.source_position.to_numpy(dtype=np.int64))
    S.npz(path.with_suffix('.model.npz'),centers=model.cluster_centers_)
    scores=external_scores(f.y.to_numpy(),labels)
    counts=np.bincount(labels,minlength=k)
    # Fixed identity-based silhouette sample; labels never determine sampling.
    import hashlib
    selected=np.asarray(sorted(range(n),key=lambda i:hashlib.sha256((str(f.row_id.iloc[i])+'|'+str(i)).encode()).digest())[:2000])
    with threadpool_limits(limits=1):
        sil=float(silhouette_score(X[selected],labels[selected])) if 1<len(np.unique(labels[selected]))<len(selected) else None
    record={**t,'phase':'complete','plan_sha256':S.sha(Path(root)/'plan.json'),'source_sha256':S.sha(__file__),
            'feature_sha256':S.sha(fp),'panel_sha256':S.sha(plan['panels'][t['dataset']]['table']),
            'n':n,'requested_clusters':k,'actual_clusters':int(np.count_nonzero(counts)),
            'algorithm':'MiniBatchKMeans','params':params,'labels_used_for_fit_or_selection':False,
            'failed_encoding_rows':int((~ok).sum()),'shared_valid_coverage':float(ok.mean()),
            'fit_seconds':fit_seconds,'elapsed_seconds':time.monotonic()-start,**scores,'silhouette':sil,
            'largest_cluster_fraction':float(counts.max()/n),'reaction_singleton_fraction':float(np.sum(counts==1)/n),
            'assignments_sha256':S.sha(path.with_suffix('.npz')),'centers_sha256':S.sha(path.with_suffix('.model.npz')),'finished_utc':S.now()}
    S.write(path,record);verify_result(root,plan,t,record)
    return True


def run(root=S.ROOT,worker=0,workers=2,threads=4):
    root=Path(root);plan=S.verify(root);tasks=plan['tasks']['clustering']
    # Deterministic disjoint shards, same five seeds at every endpoint.
    tasks=[t for i,t in enumerate(tasks) if i%workers==worker]
    while not (root/'HOLD_NEW_FITS').exists():
        pending=0;launched=False
        for t in tasks:
            if output(root,t).exists():continue
            pending+=1
            lock=root/'claims'/('clustering__'+key(t)+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
            with lock.open('a+') as f:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                S.write(root/f'cluster_worker{worker}.json',{'phase':'fitting_or_waiting','task':t,'updated_utc':S.now()})
                if fit(root,plan,t,threads):launched=True;break
        if not pending:break
        if not launched:time.sleep(20)
    S.write(root/f'cluster_worker{worker}.json',{'phase':'finished_or_held','updated_utc':S.now()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=S.ROOT);p.add_argument('--worker',type=int,choices=[0,1],required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args();run(a.root,a.worker,2,a.threads)
