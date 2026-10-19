"""Independent GPU0 checkpoint verification while the frozen GPU1 queue trains.

This supplemental audit never changes a canonical result, training provenance,
or model. The physical GPU is honestly recorded as zero throughout.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path.home()/'.local/share/synrfp-runs/final_comparison_v1'
REG=ROOT/'regression'
OUT=ROOT/'gpu0_assist'
EXPORT=HERE/'publication_metadata/final_comparison_v1/gpu0_assist'
LEASE=HERE.parent/'12_validation/runs/gpu_0.lock'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def run(timeout=600,pilot=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise RuntimeError('Select physical GPU0 only for this independent audit')
    with LEASE.open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from torch import nn
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        if torch.__version__!='2.4.0.post301':raise RuntimeError('Legacy numerical backend differs')
        plan=json.loads((REG/'plan.json').read_text())
        plan_sha=sha(REG/'plan.json')
        if plan_sha!=json.loads((REG/'plan_seal.json').read_text())['plan_sha256']:
            raise ValueError('Changed original regression plan')
        checked={};started=time.monotonic()
        jobs=plan['jobs'][:1] if pilot else plan['jobs']
        total=1 if pilot else 174
        while len(checked)<total:
            changed=False
            for task in jobs:
                path=REG/'models'/task['id']/'result.json'
                if not path.exists() or task['id'] in checked:continue
                result=json.loads(path.read_text())
                if result['plan_sha256']!=plan_sha or result['physical_gpu']!=1:
                    raise ValueError('Unexpected native training identity')
                directory=Path(result['attempt_directory'])
                for name in ('best_model.pt','predictions.npy','job.json','training.json'):
                    if sha(directory/name)!=result['artifacts'][name]:
                        raise ValueError('Changed native artifact: '+name)
                checkpoint=torch.load(directory/'best_model.pt',map_location='cpu',weights_only=False)
                job=json.loads((directory/'job.json').read_text())
                if checkpoint['job_sha256']!=sha(directory/'job.json') or checkpoint['params']!=job['params']:
                    raise ValueError('Checkpoint job binding differs')
                for name,expected in job['inputs'].items():
                    if sha(directory/name)!=expected:raise ValueError('Changed prediction input')
                model=nn.Sequential(nn.Linear(checkpoint['input_dim'],checkpoint['params']['hidden']),nn.ReLU(),
                                    nn.Linear(checkpoint['params']['hidden'],1)).cuda()
                model.load_state_dict(checkpoint['state_dict']);model.eval()
                X=np.load(directory/'test.npy',allow_pickle=False,mmap_mode='r')
                with np.load(directory/'preprocessing.npz',allow_pickle=False) as z:
                    mean,scale=float(z['target_mean']),float(z['target_scale'])
                predicted=[]
                replay_started=time.monotonic()
                with torch.no_grad():
                    for offset in range(0,len(X),256):
                        chunk=np.asarray(X[offset:offset+256]).copy()
                        prediction=model(torch.as_tensor(chunk,dtype=torch.float32,device='cuda')).squeeze(-1).cpu().numpy()
                        predicted.append(prediction.astype(np.float64)*scale+mean)
                actual=np.concatenate(predicted)
                expected=np.load(directory/'predictions.npy',allow_pickle=False)
                if not np.allclose(actual,expected,atol=1e-5,rtol=1e-5):
                    raise ValueError('Independent GPU0 checkpoint replay differs')
                checked[task['id']]=dict(passed=True,result_path=str(path),result_sha256=sha(path),
                    samples=len(X),all_heldout_rows=True,max_abs_difference=float(np.max(np.abs(actual-expected))),physical_gpu=0,
                    prediction_seconds=time.monotonic()-replay_started,chunk_size=256)
                del model,checkpoint,X,actual,expected
                changed=True
            status=dict(phase='complete' if len(checked)==total else 'running',completed=len(checked),expected=total,
                physical_gpu=0,training_physical_gpu=1,scope='supplemental independent FULL heldout checkpoint replay; no training or canonical result changes',
                original_plan_sha256=plan_sha,source_sha256=sha(__file__),torch_version=torch.__version__,
                elapsed_seconds=time.monotonic()-started,updated_utc=now())
            write(OUT/('pilot_status.json' if pilot else 'status.json'),status)
            if changed:print(json.dumps(status),flush=True)
            if time.monotonic()-started>timeout:raise RuntimeError('Bounded wait for native completion exceeded')
            if len(checked)<total:time.sleep(1)
        receipt=dict(status,checks=checked)
        write(OUT/('PILOT_READY.json' if pilot else 'COMPLETE.json'),receipt)
        write(EXPORT/('PILOT_READY.json' if pilot else 'COMPLETE.json'),receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    run(args.timeout,args.pilot)
