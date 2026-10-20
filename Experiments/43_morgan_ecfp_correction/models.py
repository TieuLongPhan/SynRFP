"""Source-bound reuse of the original study22 CUDA arithmetic and compact jobs."""
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import study as S
import encode as E

M=S.module('morgan43_original_models',S.OLD_HERE/'models.py',S.legacy_shim())
WORKER=S.OLD_HERE/'gpu_worker_runtime_v2.py'
_identity=M.identity
_prepare=M.prepare_job

def identity(root,plan,task,path):
    return {**_identity(root,plan,task,path),'adapter_source_sha256':S.sha(WORKER),
        'correction_controller_sha256':S.sha(__file__),'tokenizer_profile':S.PROFILE,'core_ready_sha256':plan['core_ready_sha256']}

def feature(root,plan,task):
    panel=plan['panels'][task['dataset']];source=plan['sources'][panel['source']]
    if E.feature_ready(root,plan,panel['source'],cache=CACHE) is None:return None
    return panel,source,Path(source['features'][task['config']])

def prepare_job(root,plan,task,path,directory):
    value=_prepare(root,plan,task,path,directory)
    job=S.read(directory/'job.json');job['adapter_source_sha256']=S.sha(WORKER);S.write(directory/'job.json',job)
    return value

M.identity=identity;M.prepare_job=prepare_job;M.feature=feature;M.output=S.result_path
source=inspect.getsource(M.fit)
S.need(source.count("S.HERE/'gpu_worker.py'")==2,'Original fit dispatch boundary changed')
source=source.replace("S.HERE/'gpu_worker.py'","S.HERE/'gpu_worker_runtime_v2.py'")
exec(compile(source,str(S.OLD_HERE/'models.py')+'[morgan43-versioned-inputs]','exec'),M.__dict__)
CACHE=S.HashCache()

def verify_result(root,plan,task,record=None,deep=False,cache=None):
    path=S.result_path(root,task)
    if record is None:
        if not path.exists():return None
        record=S.read(path)
    ready=feature(root,plan,task);S.need(ready is not None,'Missing corrected features')
    frozen=identity(root,plan,task,ready[2]);S.need(record['phase']=='complete' and record['identity']==frozen,'Changed corrected MLP identity')
    S.need(record['training_params']==plan['params']['classification'] and record['physical_gpu']==1 and record['backend']=='torch_cuda','Changed MLP recipe/device')
    M.audit_result(root,plan,task,record)
    directory=Path(record['attempt_directory']);panel=plan['panels'][task['dataset']];f=pd.read_csv(panel['table'],dtype={'row_id':str})
    code=np.load(panel['split_paths'][str(task['fold'])],allow_pickle=False)
    with np.load(directory/'rows.npz',allow_pickle=False) as z:
        S.need(np.array_equal(z['outer'],np.flatnonzero(code!=2)) and np.array_equal(z['test'],np.flatnonzero(code==2)),'Changed MLP outer selectors')
        with np.load(directory/'test_predictions.npz',allow_pickle=False) as pred:
            S.need(np.array_equal(pred['source_positions'],f.source_position.to_numpy()[z['test']]),'Changed MLP source positions')
    job=S.read(directory/'job.json');S.need(job['test_outcomes_passed_to_worker'] is False and job['adapter_source_sha256']==S.sha(WORKER),'Held-out labels or wrong arithmetic adapter')
    if deep:
        classes,state=M.initialization(f.y.to_numpy()[np.flatnonzero(code!=2)],plan['params']['classification'],task['fold'])
        with np.load(directory/'state.npz',allow_pickle=False) as z:
            S.need(set(z.files)==set(state) and all(np.array_equal(z[k],v) for k,v in state.items()),'Changed original initialization/RNG')
    return record

def fit(root,plan,task,gpu=1):
    S.need(gpu==1,'GPU0 is reserved')
    E.disk_guard(root)
    if S.result_path(root,task).exists():return verify_result(root,plan,task)
    result=M.fit(root,plan,task,gpu)
    if result:verify_result(root,plan,task,deep=True)
    return result
