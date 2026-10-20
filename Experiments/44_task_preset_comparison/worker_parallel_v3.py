"""Run the original numerical producer on one reviewed independent shard."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re

import parallel_v3 as P
C=P.module('five44_original_scientific_producer',P.HERE/'clustering.py')

def run(root=P.ROOT,worker=0,run_id=None):
    root=Path(root);plan,review=P.review(root)
    P.need(worker in range(5) and isinstance(run_id,str) and re.fullmatch(r'v3-w[0-4]-g[0-9]{4,}',run_id),
           'Unsafe or foreign five-shard invocation')
    P.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Independent clustering must hide CUDA')
    old_stage=C.stage_items;old_immutable=C.immutable
    C.stage_items=lambda scientific,index,workers=5:P.stage_items(root,scientific,index,workers)
    def immutable(path,value):
        old_immutable(path,value)
        if Path(path).name=='STARTED.json':
            out=root/'parallel_v3'
            original=root/'workers'/run_id/'OPERATIONAL.json'
            C.immutable=old_immutable
            try:
                old_immutable(original,dict(schema='task-preset44-five-shard-worker-v3',phase='ready',run_id=run_id,
                    worker=worker,workers=5,threads=1,memory_gib=7,pid=os.getpid(),
                    partition_plan_sha256=P.sha(out/'plan.json'),scientific_plan_sha256=plan['scientific_plan_sha256'],
                    sources_sha256=P.sources(),outputs_sha256={str(out/n):P.sha(out/n)for n in
                        ('plan.json','plan_seal.json','CODE_READY.json','ROOT_REVIEW.json')}))
            finally:C.immutable=immutable
    C.immutable=immutable
    try:return C.run(root,worker=worker,workers=5,threads=1,run_id=run_id)
    finally:C.stage_items=old_stage;C.immutable=old_immutable

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=P.ROOT);p.add_argument('--worker',type=int,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args();run(a.root,a.worker,a.run_id)
