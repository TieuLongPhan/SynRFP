"""Five independent CPU lanes; exact old cells and paired population shards."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=Path.home()/'.local/share/synrfp-runs/task_preset_comparison_v1'
ROOT34=ROOT.parent/'final_comparison_v1/clustering'
EXPORT34=HERE.parent/'34_final_comparison/publication_metadata/final_comparison_v1/clustering'
WORKERS=5
MEMORY_GIB=7
CORES=('18','19','26','27','28')
OWN=('parallel_v3.py','worker_parallel_v3.py','controller_parallel_v3.py','test_parallel_v3.py',
     'report_parallel_v3.py','test_report_parallel_v3.py','PARALLEL_V3.md','synrfp-task-preset44-parallel-v3.service')

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
U=module('five44_common',HERE/'common.py')
read,write,sha,need=U.read,U.write,U.sha,U.need
_CACHE=U.HashCache()
def sources():return {str((HERE/n).resolve()):sha(HERE/n) for n in OWN}
def key(task):return task['dataset'],task['population'],int(task['seed'])

def lpt(tasks,weights,workers=WORKERS):
    """The all/shared population pair is one indivisible scheduling group."""
    need(workers==WORKERS,'Exactly five independent lanes are reviewed')
    groups={}
    for task in tasks:groups.setdefault((task['dataset'],int(task['seed'])),[]).append(task)
    ordered=[]
    for group,items in groups.items():
        need(len(items)==2 and {x['population'] for x in items}=={'all_rows','shared_valid_all_six'},'Missing population pair')
        items.sort(key=lambda x:0 if x['population']=='all_rows' else 1)
        weight=sum(weights[key(t)] for t in items)
        need(math.isfinite(weight) and weight>0,'Invalid preexisting timing weight')
        ordered.append((weight,group,items))
    assignments={str(i):[] for i in range(workers)};loads={str(i):0. for i in range(workers)}
    for weight,group,items in sorted(ordered,key=lambda x:(-x[0],x[1])):
        lane=min(loads,key=lambda x:(loads[x],int(x)));assignments[lane].extend(items);loads[lane]+=weight
    return assignments,loads

def validate_partition(plan,scientific):
    need(plan.get('schema')=='task-preset44-five-shard-plan-v3' and plan.get('workers')==5
         and plan.get('threads')==1 and plan.get('memory_gib')==7,'Wrong independent-fit profile')
    need(set(plan.get('assignments',{}))==set(map(str,range(5))),'Missing five-way shard')
    expected={key(t):t for t in scientific['tasks']};seen={}
    for worker,tasks in plan['assignments'].items():
        need(len(tasks)%2==0,'Population dependency pair was split')
        for i in range(0,len(tasks),2):
            a,b=tasks[i:i+2]
            need(a['population']=='all_rows' and b['population']=='shared_valid_all_six'
                 and(a['dataset'],a['seed'])==(b['dataset'],b['seed']),'Population copy dependency changed')
        for task in tasks:
            k=key(task);need(k not in seen and expected.get(k)==task,'Duplicate/foreign/changed five-way task');seen[k]=task
    need(seen==expected and len(seen)==240,'Five-way union does not cover exact240 cells')
    need(sum(t['origin']=='fresh'for t in seen.values())==200,'Changed actual fit population')
    return plan

def prepare(root=ROOT):
    root=Path(root);out=root/'parallel_v3';path=out/'plan.json'
    if path.exists():return verify_partition(root)
    pp=root/'clustering/plan.json';scientific=read(pp)
    _CACHE.verify(pp,read(root/'clustering/plan_seal.json')['sha256'])
    complete=read(ROOT34/'COMPLETE.json');csvpath=EXPORT34/'cells.csv'
    need(complete.get('phase')=='complete' and complete.get('completed')==complete.get('expected')==240
         and complete.get('all_artifact_hashes_verified') is True and complete.get('full_metric_replay') is True,
         'Native timing/memory catalogue is not complete')
    _CACHE.verify(csvpath,complete['outputs_sha256']['cells.csv'])
    with csvpath.open(newline='')as f:rows=list(csv.DictReader(f))
    wanted={key(t)for t in scientific['tasks']};need(len(rows)==240 and{key(r)for r in rows}==wanted,'Wrong native cost grid')
    pins={str(pp):sha(pp),str(root/'clustering/plan_seal.json'):sha(root/'clustering/plan_seal.json'),
          str(ROOT34/'COMPLETE.json'):sha(ROOT34/'COMPLETE.json'),str(csvpath):sha(csvpath)}
    weights={};peak=0.;fresh=0
    for row in rows:
        p=Path(row['result_path']);_CACHE.verify(p,row['result_sha256']);r=read(p)
        need(key(r)==key(row) and r['plan_sha256']==complete['plan_sha256'],'Native timing identity changed')
        weight=float(r['elapsed_seconds']);need(math.isfinite(weight) and weight>=0,'Undefined native timing')
        weights[key(row)]=max(weight,1e-9)
        if r['origin']=='fresh':
            fresh+=1;peak=max(peak,float(r['peak_rss_gib']))
        pins[str(p.resolve())]=row['result_sha256']
    need(fresh==200 and 0<peak<7,'Native full-cell memory evidence exceeds new seven-GiB envelope')
    assignments,loads=lpt(scientific['tasks'],weights)
    plan=dict(schema='task-preset44-five-shard-plan-v3',scientific_plan_sha256=sha(pp),
              configuration=scientific['config'],workers=5,threads=1,memory_gib=7,cores=list(CORES),
              assignments=assignments,estimated_native_seconds=loads,counts=dict(cells=240,fresh=200,copies=40,pairs=120),
              memory_evidence=dict(native_fresh_fits=200,max_native_peak_rss_gib=peak,width=2048,
                  same_unscaled_float32_geometry=True,extra_memory_above_observed_fraction=7/peak-1),
              inputs_sha256=pins,sources_sha256=sources(),gpu_jobs=0,
              scheduling='Longest historical native elapsed population-pair first; immutable five-lane deterministic LPT')
    validate_partition(plan,scientific);U.immutable(path,plan);U.immutable(out/'plan_seal.json',dict(sha256=sha(path)))
    return verify_partition(root)

def verify_partition(root=ROOT):
    root=Path(root);out=root/'parallel_v3';path=out/'plan.json'
    _CACHE.verify(path,read(out/'plan_seal.json')['sha256']);p=read(path)
    _CACHE.verify(root/'clustering/plan.json',p['scientific_plan_sha256'])
    need(p['sources_sha256']==sources(),'Changed five-shard operational source')
    U.verify_catalog(p['sources_sha256'],_CACHE);U.verify_catalog(p['inputs_sha256'],_CACHE)
    return validate_partition(p,read(root/'clustering/plan.json'))

def review(root=ROOT):
    root=Path(root);out=root/'parallel_v3';plan=verify_partition(root)
    ready=read(out/'CODE_READY.json');r=read(out/'ROOT_REVIEW.json')
    need(ready.get('phase')=='complete' and ready.get('tests_passed',0)>0 and ready.get('sources_sha256')==sources(),
         'Five-shard code readiness is absent or changed')
    need(r.get('schema')=='task-preset44-five-shard-root-reviewed' and r.get('phase')=='reviewed'
         and r.get('plan_sha256')==sha(out/'plan.json') and r.get('code_ready_sha256')==sha(out/'CODE_READY.json')
         and r.get('workers')==5 and r.get('worker_cpu')==r.get('worker_threads')==1
         and r.get('worker_memory_gib')==7 and r.get('gpu_jobs')==0,'Root five-shard review differs')
    need(all(r.get('sources_sha256',{}).get(p)==h for p,h in sources().items()),'Root review omits five-shard sources')
    U.verify_catalog(r['sources_sha256'],_CACHE)
    return plan,r

def stage_items(root,scientific,worker,workers=5):
    need(workers==5 and worker in range(5),'Only the exact five-way worker cohort is admitted')
    plan=verify_partition(root);need(sha(Path(root)/'clustering/plan.json')==plan['scientific_plan_sha256'],'Scientific plan differs')
    return plan['assignments'][str(worker)]
