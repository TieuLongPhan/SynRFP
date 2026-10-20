"""Versioned five-CPU continuation of the unchanged Path comparison."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
O=module('parallel44_original_controller',HERE/'controller.py')
ROOT=O.ROOT
CONTROL=ROOT/'control_parallel_v2'
REVIEW_ROOT=ROOT/'parallel_v2'
ORIGINAL_UNIT='synrfp-task-preset44-v1.service'
CORES='18,19,26,27,28'
CPU=5
MEMORY_GIB=12
ORIGINAL_WORKER_SPEC=O.worker_spec
ORIGINAL_ADMISSION=O.admission
ADMITTED=None

def sources():
    names=('controller_parallel_v2.py','test_controller_parallel_v2.py','PARALLEL_V2.md','synrfp-task-preset44-parallel-v2.service')
    return {str((HERE/n).resolve()):O.B.sha(HERE/n) for n in names}

def admission(value,cpu,memory):
    # The original cycle's precheck describes its one-CPU worker. Account for
    # the new five-CPU worker before publishing a fresh dispatch. C.ensure also
    # independently admits the actual job's five-CPU finite cgroup.
    if cpu==1 and memory==MEMORY_GIB*O.GIB:cpu=CPU
    return ORIGINAL_ADMISSION(value,cpu,memory)
O.admission=admission

def manifest():
    global ADMITTED
    paths={n:REVIEW_ROOT/n for n in ('CODE_READY.json','ROOT_REVIEW.json')}
    if not all(p.exists() for p in paths.values()):return None
    ready,review=(O.read(paths[n]) for n in paths)
    O.need(ready.get('phase')=='complete' and ready.get('tests_passed',0)>0
           and ready.get('sources_sha256')==sources(),'Five-CPU adapter source/readiness changed')
    O.pins(ready['sources_sha256']);O.pins(review.get('sources_sha256'))
    O.need(review.get('schema')=='task-preset44-parallel-v2-root-reviewed' and review.get('phase')=='reviewed'
           and review.get('worker_cpu')==CPU and review.get('worker_threads')==CPU
           and review.get('worker_memory_gib')==MEMORY_GIB and review.get('worker_cores')==CORES
           and review.get('old_worker_yield_required') is True and review.get('gpu_jobs')==0
           and review.get('code_ready_sha256')==O.B.sha(paths['CODE_READY.json'])
           and review.get('original_execution_sha256')==O.B.sha(ROOT/'control/execution.json'),
           'Root review does not bind the exact five-CPU continuation')
    original=O.manifest()
    if original is None:return None
    required={**original['sources_sha256'],**ready['sources_sha256']}
    O.need(all(review['sources_sha256'].get(p)==h for p,h in required.items()),'Parallel review omits original or new source')
    props=O.C.props(ORIGINAL_UNIT)
    O.need(props.get('ActiveState') not in O.C.ACTIVE and props.get('MainPID','0')=='0','Original44 governor must exit before takeover')
    value=dict(schema='task-preset44-parallel-execution-v2',phase='ready',
        original_execution_sha256=O.B.sha(ROOT/'control/execution.json'),
        sources_sha256={**required,str(paths['CODE_READY.json']):O.B.sha(paths['CODE_READY.json']),
                        str(paths['ROOT_REVIEW.json']):O.B.sha(paths['ROOT_REVIEW.json'])},
        worker_cpu=CPU,worker_threads=CPU,worker_memory_gib=MEMORY_GIB,worker_cores=CORES,
        cpu_policy_fraction=.85,authorized_memory_ceiling_fraction=.90,effective_stricter_memory_fraction=.85,
        original_lock=str(ROOT/'control/controller.lock'),scientific_sources_and_plan_unchanged=True,
        gpu_jobs=0,old_results_and_thread_counts_preserved=True)
    O.immutable(CONTROL/'execution.json',value)
    ADMITTED=value
    return value

def worker_spec(generation,execution):
    O.need(ADMITTED is not None,'Five-CPU root review is not admitted')
    # The original cycle advances the generation only after the old terminal
    # receipt closes and its service exits. Keep that proof explicit here.
    plan=O.read(ROOT/'clustering/plan.json')
    for job in O.dispatched():
        if job.get('stage')!='clustering':continue
        p=O.C.props(job['unit'])
        O.need(p.get('ActiveState') not in O.C.ACTIVE and p.get('MainPID','0')=='0',
               'A previous44 scientific invocation is still live')
        O.need(O.terminal(job,execution,plan) in ('yielded','complete'),'Previous44 invocation has no audited terminal receipt')
    job=ORIGINAL_WORKER_SPEC(generation,execution)
    job['cpu_equivalents']=CPU;job['memory_gib']=MEMORY_GIB;job['cores']=CORES
    args=job['arguments'];args[args.index('--threads')+1]=str(CPU)
    job['environment'].update(OMP_NUM_THREADS=str(CPU),MKL_NUM_THREADS=str(CPU),OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    job['sources_sha256']={**job['sources_sha256'],**ADMITTED['sources_sha256'],
                           str(CONTROL/'execution.json'):O.B.sha(CONTROL/'execution.json')}
    return job
O.worker_spec=worker_spec

def cycle():
    if manifest() is None:return dict(phase='waiting_reviewed_parallel_continuation',gpu_jobs=0)
    result=O.cycle()
    O.write(O.CONTROL/'status.json',{**result,'operational_adapter':'parallel-v2','updated_utc':O.B.now()})
    return result

def preflight():
    p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_controller_parallel_v2.py','-v'],capture_output=True,text=True)
    O.need(p.returncode==0,'Parallel44 controls failed: '+p.stdout+p.stderr)
    m=re.search(r'Ran (\d+) tests',p.stderr);O.need(m is not None,'No executed parallel44 controls')
    v=dict(schema='task-preset44-parallel-code-ready-v2',phase='complete',tests_passed=int(m.group(1)),sources_sha256=sources())
    return O.immutable(REVIEW_ROOT/'CODE_READY.json',v)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    if a.preflight:print(json.dumps(preflight()));return
    with (ROOT/'control/controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (CONTROL/'FAILED.json').exists() or (ROOT/'control/FAILED.json').exists():return
        try:
            while True:
                value={**cycle(),'updated_utc':O.B.now()};O.write(CONTROL/'status.json',value)
                if a.once or value.get('phase')=='complete':return
                time.sleep(30)
        except Exception as error:
            # New operational review errors stop dispatch; no scientific worker
            # is killed or signalled by this handoff's error path.
            O.write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),
                    scientific_artifacts_preserved=True,scientific_signals_sent=0,updated_utc=O.B.now()))
            raise

if __name__=='__main__':main()
