"""Full-row matched GPU fits with compact, replayable inputs and predictions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import matthews_corrcoef,accuracy_score,f1_score,mean_absolute_error,mean_squared_error,r2_score

import study as S

GPU_PYTHON='/home/labhhc4/anaconda3/envs/synprop/bin/python'


def initialization(y,params,seed):
    lb=LabelBinarizer().fit(y);target=lb.transform(y).astype(bool)
    if len(lb.classes_)<2:raise ValueError('Training population lacks two classes')
    clf=MLPClassifier(early_stopping=True,random_state=seed)
    clf._label_binarizer=lb;clf._random_state=np.random.RandomState(seed)
    clf._initialize(target,[2048,params['hidden'][0],target.shape[1]],np.dtype('float32'))
    tr,va=train_test_split(np.arange(len(y)),test_size=.1,random_state=clf._random_state,
                          stratify=target if target.shape[1]==1 else None)
    state=clf._random_state.get_state()
    return lb.classes_,{'train':tr,'val':va,'y':np.searchsorted(lb.classes_,y),
            'w0':clf.coefs_[0],'w1':clf.coefs_[1],'b0':clf.intercepts_[0],'b1':clf.intercepts_[1],
            'rng_keys':state[1],'rng_pos':state[2],'rng_has_gauss':state[3],'rng_gauss':state[4]}


def task_id(task):
    repeat='f'+str(task['fold']) if task['task']=='classification' else task['split']
    return task['task']+'__'+task['dataset']+'__'+repeat+'__'+task['config']


def output(root,task):return Path(root)/'models'/task_id(task)/'result.json'


def metrics(task,truth,pred):
    if len(truth)!=len(pred) or not np.isfinite(pred).all():raise ValueError('Invalid held-out prediction shape or values')
    if task=='classification':
        return {'mcc':float(matthews_corrcoef(truth,pred)),'accuracy':float(accuracy_score(truth,pred)),
                'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0))}
    return {'r2':float(r2_score(truth,pred)),'mae':float(mean_absolute_error(truth,pred)),
            'rmse':float(np.sqrt(mean_squared_error(truth,pred)))}


def feature(root,plan,task):
    panel=plan['panels'][task['dataset']];source=plan['sources'][panel['source']]
    receipt=Path(source['feature_receipt'])
    if not receipt.exists():return None
    value=json.loads(receipt.read_text());path=Path(source['features'][task['config']])
    if value['plan_sha256']!=S.sha(Path(root)/'plan.json'):raise ValueError('Wrong full-row feature plan')
    for p in [path,path.with_suffix('.json')]:
        if value['files'][p.name]!=S.sha(p):raise ValueError('Changed feature artifact')
    meta=json.loads(path.with_suffix('.json').read_text())
    if meta['spec']!=plan['configs'][task['config']] or meta['source_table_sha256']!=S.sha(source['table']):raise ValueError('Changed feature contract')
    return panel,source,path


def identity(root,plan,task,path):
    panel=plan['panels'][task['dataset']]
    split=panel['split_paths'][str(task['fold'])] if task['task']=='classification' else next(
        s['path'] for s in plan['yield_splits'] if s['dataset']==task['dataset'] and s['split']==task['split'])
    return {**task,'plan_sha256':S.sha(Path(root)/'plan.json'),'panel_sha256':S.sha(panel['table']),
            'split_sha256':S.sha(split),'feature_sha256':S.sha(path),
            'adapter_source_sha256':S.sha(S.HERE/'gpu_worker.py'),'controller_source_sha256':S.sha(__file__)}


def complete(root,plan,task,frozen=None,audit=True):
    p=output(root,task)
    if not p.exists():return None
    r=json.loads(p.read_text())
    if r.get('phase')!='complete' or r['identity']['plan_sha256']!=S.sha(Path(root)/'plan.json'):raise ValueError('Changed completed model')
    if frozen is not None and r['identity']!=frozen:raise ValueError('Different completed model identity')
    if audit:
        audit_result(root,plan,task,r)
    return r


def prepare_job(root,plan,task,path,directory):
    panel=plan['panels'][task['dataset']];f=pd.read_csv(panel['table'],dtype={'row_id':str})
    is_yield=task['task']=='yield';feature_rows=np.arange(len(f),dtype=np.int64) if is_yield else f.source_position.to_numpy(dtype=np.int64)
    frozen=identity(root,plan,task,path);p=plan['params'][task['task']]
    if not is_yield:
        code=np.load(panel['split_paths'][str(task['fold'])],allow_pickle=False)
        if code.shape!=(len(f),) or not np.isin(code,[0,1,2]).all():raise ValueError('Invalid original grouped fold')
        outer,test=np.flatnonzero(code!=2),np.flatnonzero(code==2)
        if not len(outer) or not len(test):raise ValueError('Empty full grouped partition')
        seed=0 if panel['family']=='activity' else task['fold']
        classes,state=initialization(f.y.to_numpy()[outer],p,seed)
        S.npz(directory/'state.npz',**state)
        S.npz(directory/'selection.npz',outer=feature_rows[outer],test=feature_rows[test])
        S.npz(directory/'preprocessing.npz',classes=classes,mu=np.float32(0),sd=np.float32(1))
        selectors={'outer':outer,'test':test,'internal_train':outer[state['train']],'internal_validation':outer[state['val']]}
        names=['state.npz','selection.npz','preprocessing.npz','rows.npz']
    else:
        spec=next(s for s in plan['yield_splits'] if s['dataset']==task['dataset'] and s['split']==task['split'])
        with np.load(spec['path'],allow_pickle=False) as z:selectors={k:z[k] for k in ['train','validation','test']}
        if not np.array_equal(np.sort(np.concatenate(list(selectors.values()))),np.arange(len(f))):raise ValueError('Invalid full yield partitions')
        y=f.yield_pct.to_numpy(dtype=float);mean=float(y[selectors['train']].mean());scale=float(y[selectors['train']].std());scale=scale if scale>=1e-12 else 1.
        S.npz(directory/'targets.npz',y_train=((y[selectors['train']]-mean)/scale).astype(np.float32),y_val=((y[selectors['validation']]-mean)/scale).astype(np.float32))
        S.npz(directory/'selection.npz',**{k:feature_rows[v] for k,v in selectors.items()})
        S.npz(directory/'preprocessing.npz',target_mean=np.float64(mean),target_scale=np.float64(scale),
              feature_mean=np.zeros(2048,np.float32),feature_scale=np.ones(2048,np.float32))
        names=['targets.npz','selection.npz','preprocessing.npz','rows.npz'];classes=None
    S.npz(directory/'rows.npz',**{k:np.asarray(v,dtype=np.int64) for k,v in selectors.items()})
    job={'id':task_id(task),'task':task['task'],'dataset':task['dataset'],'split':task.get('split','f'+str(task.get('fold',0))),
         'fingerprint':task['config'],'params':p,'identity':frozen,'plan_sha256':frozen['plan_sha256'],
         'feature_path':str(path),'feature_sha256':frozen['feature_sha256'],
         'adapter_source_sha256':S.sha(S.HERE/'gpu_worker.py'),
         'inputs_sha256':{n:S.sha(directory/n) for n in names},'test_outcomes_passed_to_worker':False}
    S.write(directory/'job.json',job)
    return f,selectors,classes,frozen


def fit(root,plan,task,gpu):
    root=Path(root);ready=feature(root,plan,task)
    if ready is None:return False
    _,_,path=ready;frozen=identity(root,plan,task,path)
    if complete(root,plan,task,frozen):return True
    if shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Below35GiB storage reserve; stop dispatch, retain models')
    parent=output(root,task).parent;parent.mkdir(parents=True,exist_ok=True)
    directory=parent/('attempt-%03d'%(len(list(parent.glob('attempt-*')))+1));directory.mkdir()
    frame,rows,classes,frozen=prepare_job(root,plan,task,path,directory)
    S.write(root/f'gpu{gpu}_status.json',{'phase':'training','task':task,'started_utc':S.now()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    start=time.monotonic();log=root/'logs'/f'gpu{gpu}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',buffering=1) as out:
        subprocess.run([GPU_PYTHON,'-u',str(S.HERE/'gpu_worker.py'),str(directory)],env=env,stdout=out,stderr=out,check=True)
    trained=json.loads((directory/'training.json').read_text())
    if trained['job_sha256']!=S.sha(directory/'job.json') or trained['adapter_source_sha256']!=S.sha(S.HERE/'gpu_worker.py'):raise ValueError('Invalid worker provenance')
    test=rows['test'];pred=np.load(directory/'predictions.npy',allow_pickle=False)
    if task['task']=='classification':
        if not np.issubdtype(pred.dtype,np.integer) or np.any(pred<0) or np.any(pred>=len(classes)):raise ValueError('Invalid encoded class prediction')
        pred=classes[pred];truth=frame.y.to_numpy()[test]
    else:truth=frame.yield_pct.to_numpy()[test]
    values=metrics(task['task'],truth,pred)
    S.npz(directory/'test_predictions.npz',observed=truth,predictions=pred,
          source_positions=frame.source_position.to_numpy()[test],row_ids=frame.row_id.to_numpy(dtype=str)[test])
    with np.load(path,allow_pickle=False) as z:
        feature_rows=test if task['task']=='yield' else frame.source_position.to_numpy()[test]
        failed=int((~z['ok'][feature_rows]).sum())
    names=[p.name for p in directory.iterdir() if p.is_file()]
    record={'phase':'complete',**task,'identity':frozen,**values,'family':plan['panels'][task['dataset']]['family'],
             'fit_seconds':trained.get('fit_seconds',trained.get('seconds')),'elapsed_seconds':time.monotonic()-start,
             'epochs':trained['epochs'],'n':len(frame),'n_test':len(test),'n_unfeaturised_test':failed,
             'physical_gpu':gpu,'backend':'torch_cuda','completed_utc':S.now(),'training_params':plan['params'][task['task']],
             'artifacts_sha256':{n:S.sha(directory/n) for n in names},'attempt_directory':str(directory)}
    # Hashes are relative to the retained attempt, not duplicated model inputs.
    S.write(parent/'result.json',record);audit_result(root,plan,task,record)
    return True


def audit_result(root,plan,task,r):
    directory=Path(r['attempt_directory'])
    for n,h in r['artifacts_sha256'].items():
        if S.sha(directory/n)!=h:raise ValueError('Changed compact model artifact')
    with np.load(directory/'test_predictions.npz',allow_pickle=False) as z:
        frame=pd.read_csv(plan['panels'][task['dataset']]['table'],dtype={'row_id':str})
        with np.load(directory/'rows.npz',allow_pickle=False) as rows:test=rows['test']
        truth=frame.y.to_numpy()[test] if task['task']=='classification' else frame.yield_pct.to_numpy()[test]
        if not np.array_equal(z['observed'],truth) or not np.array_equal(z['row_ids'].astype(str),frame.row_id.to_numpy(dtype=str)[test]):raise ValueError('Changed prediction row/outcome identity')
        for k,v in metrics(task['task'],truth,z['predictions']).items():
            if not np.isclose(r[k],v,atol=1e-12,rtol=0):raise ValueError('Invalid compact model metrics')


def run(root=S.ROOT,gpu=1):
    root=Path(root);plan=S.verify(root)
    lease=S.HERE.parent/'12_validation/runs'/f'gpu_{gpu}.lock'
    with lease.open('a+') as device_lock:
        fcntl.flock(device_lock,fcntl.LOCK_EX)
        tasks=plan['tasks']['classification']+plan['tasks']['yield']
        # Dataset size is a runtime choice only; no scores determine job order.
        tasks=sorted(tasks,key=lambda t:(plan['panels'][t['dataset']]['n'],t['config'],task_id(t)))
        while not (root/'HOLD_NEW_FITS').exists():
            pending=0;launched=False
            for task in tasks:
                if output(root,task).exists():
                    r=json.loads(output(root,task).read_text())
                    if r['identity']['plan_sha256']!=S.sha(root/'plan.json'):raise ValueError('Old full-row model identity')
                    continue
                pending+=1;lock=root/'claims'/(task_id(task)+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
                with lock.open('a+') as claim:
                    try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if output(root,task).exists():continue
                    if feature(root,plan,task) is None:continue
                    fit(root,plan,task,gpu);launched=True;break
            if not pending:break
            if not launched:time.sleep(20)
    S.write(root/f'gpu{gpu}_status.json',{'phase':'finished_or_held','updated_utc':S.now()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=S.ROOT);p.add_argument('--gpu',type=int,choices=[0,1],required=True);a=p.parse_args();run(a.root,a.gpu)
