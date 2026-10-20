"""Shared packed inputs for unchanged classification/yield training arithmetic.

Source adapters alter input loading, device dispatch and backend provenance.
Initialization, optimizer updates, validation stopping and best-weight restore
come from the pinned original workers. No held-out outcomes enter this worker.
"""
import argparse
import ast
import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

import study as S


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def checked_job(directory):
    job=json.loads((directory/'job.json').read_text())
    for name,digest in job['inputs_sha256'].items():
        if Path(name).name!=name or S.sha(directory/name)!=digest:raise ValueError('Changed compact job input')
    path=Path(job['feature_path'])
    if S.sha(path)!=job['feature_sha256']:raise ValueError('Changed shared packed feature pool')
    if job['adapter_source_sha256']!=S.sha(__file__):raise ValueError('Changed GPU input adapter')
    with np.load(directory/'selection.npz',allow_pickle=False) as z:selection={k:z[k] for k in z.files}
    expected={'outer','test'} if job['task']=='classification' else {'train','validation','test'}
    if set(selection)!=expected:raise ValueError('Unexpected selector fields; held-out outcomes prohibited')
    return job,path,selection


def dense(path,indices):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'X','ok','n_bits','source_positions','row_ids'}:raise ValueError('Feature pool must contain features and identities only')
        if int(z['n_bits'])!=2048 or z['X'].dtype!=np.uint8 or z['X'].shape[1]!=256:raise ValueError('Wrong packed feature shape')
        if indices.dtype!=np.int64 or np.any(indices<0) or np.any(indices>=len(z['X'])):raise ValueError('Invalid source row selectors')
        x=z['X'][indices]
    return np.unpackbits(x,axis=1,count=2048).astype(np.float32)


def adapt_classification():
    path=S.HERE.parent/'16_modes/classification_worker.py'
    m=module('full_factorial_classification_source',path)
    original=inspect.getsource(m.train)
    old='if device != "cpu":\n        raise ValueError("This separately registered worker supports CPU only")'
    if original.count(old)!=1:raise ValueError('Classification source adapter boundary changed')
    source=original.replace(old,'if device not in ("cpu", "cuda"):\n        raise ValueError("Unsupported full-factorial device")')
    source=source.replace('"backend": "torch_cpu", "protocol": "sklearn_adam_fp32_cpu_v1"',
                          '"backend": "torch_" + device, "protocol": "sklearn_adam_fp32_" + device + "_v1"')
    def load_matrix(directory,stem):
        job,path,rows=checked_job(Path(directory))
        return dense(path,rows['outer'] if stem=='train' else rows['test'])
    m.load_matrix=load_matrix
    exec(compile(source,str(path)+'[shared-input-device-adapter]','exec'),m.__dict__)
    return m


def adapt_yield(device):
    path=S.HERE.parent/'13_yield/yield_retrain_cpu.py'
    m=module('full_factorial_yield_source',path)
    source=inspect.getsource(m.train)
    guard="if os.environ.get('CUDA_VISIBLE_DEVICES') != '':\n        raise RuntimeError('CPU development worker requires CUDA hidden')"
    if source.count(guard)!=1 or source.count("device = torch.device('cpu')")!=1:raise ValueError('Yield adapter boundary changed')
    source=source.replace(guard,'if FULL_DEVICE == "cuda" and not __import__("torch").cuda.is_available():\n        raise RuntimeError("CUDA unavailable")')
    source=source.replace("device = torch.device('cpu')","device = torch.device(FULL_DEVICE)")
    source=source.replace("'physical_gpu': None, 'execution_device': 'cpu'",
                          "'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'execution_device': FULL_DEVICE")
    m.FULL_DEVICE=device
    def inputs(directory,job):
        _,path,rows=checked_job(Path(directory))
        with np.load(directory/'targets.npz',allow_pickle=False) as z:
            if set(z.files)!={'y_train','y_val'}:raise ValueError('Only train/validation targets are allowed')
            targets={k:z[k] for k in z.files}
        fit={'X_train':dense(path,rows['train']),'X_val':dense(path,rows['validation']),**targets}
        for k in ['train','val']:
            if fit['y_'+k].shape!=(len(fit['X_'+k]),) or not np.isfinite(fit['y_'+k]).all():raise ValueError('Malformed fit outcomes')
        return fit,dense(path,rows['test'])
    m.inputs=inputs
    exec(compile(source,str(path)+'[shared-input-device-adapter]','exec'),m.__dict__)
    return m


def execute(directory,device='cuda'):
    directory=Path(directory);job,_,_=checked_job(directory)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is required for production full-row fits')
    if job['task']=='classification':
        result=adapt_classification().train(directory,device)
        S.write(directory/'training.json',{**result,'adapter_source_sha256':S.sha(__file__),'job_sha256':S.sha(directory/'job.json'),
                'artifacts':{n:S.sha(directory/n) for n in ['predictions.npy','best_weights.npz']}})
    elif job['task']=='yield':
        adapt_yield(device).train(directory)
        result=json.loads((directory/'training.json').read_text())
        result['adapter_source_sha256']=S.sha(__file__);S.write(directory/'training.json',result)
    else:raise ValueError('Unknown model task')
    print(json.dumps({'task':job['task'],'id':job['id'],'device':device,'training':result}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    execute(a.directory,a.device)

