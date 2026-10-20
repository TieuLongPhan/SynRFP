"""Five independent admitted fits; preserve old scientific plans and receipts."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
P=module('five44_controller_partition',HERE/'parallel_v3.py')
O=module('five44_original_operations',HERE/'controller.py')
ROOT=O.ROOT;CONTROL=ROOT/'control_parallel_v3'
C,B=O.C,O.B
C.ROOT=CONTROL/'managed';B.ROOT=C.ROOT
D=C.Digests()

def manifest():
    folder=ROOT/'parallel_v3'
    if not all((folder/n).exists()for n in ('plan.json','plan_seal.json','CODE_READY.json','ROOT_REVIEW.json')):return None
    plan,review=P.review(ROOT);original=O.manifest()
    if original is None:return None
    O.need(review.get('original_execution_sha256')==B.sha(ROOT/'control/execution.json'), 'Five-way review omits original execution')
    O.need(all(review['sources_sha256'].get(p)==h for p,h in original['sources_sha256'].items()),'Five-way review omits original source')
    for unit in ('synrfp-task-preset44-v1.service','synrfp-task-preset44-parallel-v2.service'):
        v=C.props(unit)
        O.need(v.get('ActiveState') not in C.ACTIVE and v.get('MainPID','0')=='0','Previous44 governor remains active')
    value=dict(schema='task-preset44-five-shard-execution-v3',phase='ready',
        original_execution_sha256=B.sha(ROOT/'control/execution.json'),partition_plan_sha256=B.sha(folder/'plan.json'),
        scientific_plan_sha256=plan['scientific_plan_sha256'],workers=5,worker_cpu=1,worker_memory_gib=7,
        sources_sha256={**original['sources_sha256'],**P.sources(),**{str(folder/n):B.sha(folder/n)for n in
            ('plan.json','plan_seal.json','CODE_READY.json','ROOT_REVIEW.json')}},
        gpu_jobs=0,original_lock=str(ROOT/'control/controller.lock'))
    O.immutable(CONTROL/'execution.json',value)
    return plan,value

def old_workers_done():
    # The old single-worker cohorts finish their current cell and publish a
    # terminal before independent shards reuse any retained result.
    scientific=O.read(ROOT/'clustering/plan.json')
    execution=O.read(ROOT/'control/clustering_execution.json')
    for job in O.dispatched():
        if job['stage']!='clustering':continue
        v=C.props(job['unit'])
        if v.get('ActiveState') in C.ACTIVE or v.get('MainPID','0')!='0':return False
        O.need(O.terminal(job,execution,scientific) in ('yielded','complete'),'Old44 cohort lacks audited terminal')
    return True

def job_spec(worker,generation,execution):
    run='v3-w'+str(worker)+'-g'+str(generation).zfill(4)
    return dict(id='preset44-'+run,unit='synrfp-preset44-'+run+'-'+execution['partition_plan_sha256'][:8],python=O.PYTHON,
        arguments=[str(HERE/'worker_parallel_v3.py'),'--root',str(ROOT),'--worker',str(worker),'--run-id',run],
        cpu_equivalents=1,memory_gib=7,cores=P.CORES[worker],sources_sha256=execution['sources_sha256'],
        stage='clustering',worker=worker,workers=5,run_id=run,generation=generation,
        environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'),
        work_done=dict(kind='receipt',path=str(CONTROL/'closed_runs'/(run+'.json')),equals=dict(phase='complete',run_id=run)))

def publish(job):
    path=CONTROL/'dispatch'/(job['run_id']+'.json');O.immutable(path,job)
    O.immutable(CONTROL/'dispatch_seals'/path.name,dict(sha256=B.sha(path)))

def dispatched():
    jobs=[]
    for path in sorted((CONTROL/'dispatch').glob('*.json')):
        D.verify(path,O.read(CONTROL/'dispatch_seals'/path.name)['sha256']);job=O.read(path)
        O.need(re.fullmatch(r'v3-w[0-4]-g[0-9]{4,}',job['run_id']) is not None,'Foreign five-way dispatch')
        O.pins(job['sources_sha256']);C.attest_service(job,C.props(job['unit']));jobs.append(job)
    return jobs

def validate_terminal(path,job,partition,execution):
    r=O.read(path);kind=r.get('phase');scientific=O.read(ROOT/'clustering/plan.json')
    wanted={str(ROOT/'clustering/cells'/t['dataset']/t['population']/('s'+str(t['seed'])+'.json'))
            for t in partition['assignments'][str(job['worker'])]}
    O.need(kind in ('complete','yielded') and r.get('stage')=='clustering' and r.get('run_id')==job['run_id']
        and r.get('worker')==job['worker'] and r.get('workers')==5
        and r.get('plan_sha256')==execution['scientific_plan_sha256']
        and r.get('sources_sha256')==scientific['sources_sha256'],'Wrong five-way scientific closure')
    catalog=r.get('outputs_sha256',{})
    O.need(isinstance(catalog,dict) and set(catalog)<=wanted and r.get('expected_items')==len(wanted)
        and r.get('audited_items')==len(catalog),'Five-way shard count/catalog mismatch')
    O.need((kind=='complete')==(set(catalog)==wanted)==r.get('all_shard_items_verified'),'Five-way terminal completeness differs')
    ops=ROOT/'workers'/job['run_id']/'OPERATIONAL.json';v=O.read(ops)
    O.need(v.get('phase')=='ready' and v.get('worker')==job['worker'] and v.get('workers')==5
        and v.get('run_id')==job['run_id'] and v.get('threads')==1 and v.get('memory_gib')==7
        and v.get('partition_plan_sha256')==execution['partition_plan_sha256']
        and v.get('scientific_plan_sha256')==execution['scientific_plan_sha256']
        and v.get('sources_sha256')==P.sources(),'Five-way operational worker provenance differs')
    O.pins(v['outputs_sha256']);O.pins(v['sources_sha256'])
    for p,h in catalog.items():D.verify(p,h);O.tree_audit(p)
    return kind

def terminal(job,partition,execution):
    folder=ROOT/'workers'/job['run_id'];paths=[p for p in (folder/'COMPLETE.json',folder/'YIELDED.json')if p.exists()]
    O.need(len(paths)<2,'Ambiguous five-way terminal')
    if not paths:return None
    path=paths[0];kind=validate_terminal(path,job,partition,execution)
    O.need(kind==('complete'if path.name=='COMPLETE.json'else'yielded'),'Five-way terminal filename differs')
    O.immutable(CONTROL/'closed_runs'/(job['run_id']+'.json'),dict(phase='complete',run_id=job['run_id'],kind=kind,
        outputs_sha256={str(path):B.sha(path),str(folder/'OPERATIONAL.json'):B.sha(folder/'OPERATIONAL.json')}))
    return kind

def request_yield(job):
    O.need(job.get('id')=='preset44-'+job.get('run_id','') and re.fullmatch(r'v3-w[0-4]-g[0-9]{4,}',job.get('run_id','')),
           'May signal only an owned five-way invocation')
    v=C.props(job['unit']);C.attest_service(job,v)
    if v.get('ActiveState') not in C.ACTIVE:return False
    path=ROOT/'workers'/job['run_id']/'STARTED.json'
    if not path.exists():return False
    r=O.read(path)
    if str(r.get('pid'))!=v.get('MainPID'):return False
    O.need(r.get('phase')=='ready' and r.get('run_id')==job['run_id'] and r.get('workers')==5,'Wrong five-way yield handler')
    subprocess.run(['systemctl','--user','kill','--kill-whom=main','--signal=SIGUSR1',job['unit']],check=True)
    return True

def reservation_visible(job):
    for _ in range(5):
        units={x['unit'].removesuffix('.service'):x for x in O.snapshot()['units']}
        u=units.get(job['unit'])
        if u:
            O.need(u['cpu_equivalents']==1 and u['memory_max_bytes']==7*O.GIB,'Five-way cgroup cap differs')
            C.attest_service(job,C.props(job['unit']));return True
        time.sleep(.1)
    raise C.IntegrityError('Five-way cgroup reservation is not visible')

def worker_complete(index,jobs,partition,execution):
    path=ROOT/('clustering_worker_'+str(index)+'_COMPLETE.json')
    if not path.exists():return False
    r=O.read(path);matches=[j for j in jobs if j['run_id']==r.get('run_id')]
    O.need(len(matches)==1,'Unknown five-way completed invocation')
    O.need(validate_terminal(path,matches[0],partition,execution)=='complete','Incomplete five-way shard')
    return True

def latest(index,jobs,partition,execution):
    previous=[j for j in jobs if j['worker']==index]
    if not previous:return None,0,None
    job=previous[-1];kind=terminal(job,partition,execution);v=C.props(job['unit'])
    if v.get('ActiveState') in C.ACTIVE:return job,job['generation'],None
    if kind:return None,job['generation'],None
    invoked=bool(v.get('InvocationID'))or(ROOT/'workers'/job['run_id']).exists()
    if not invoked:return job,job['generation'],None
    marker=CONTROL/'abandoned_runs'/(job['run_id']+'.json')
    if not marker.exists():
        logs=C.failure_logs(job,v,{})
        O.immutable(marker,dict(phase='abandoned',worker=index,reason=C.fatal_reason(v.get('Result'),logs),
            retry_after_epoch=time.time()+30,logs=logs,artifacts_preserved=True))
    failed=O.read(marker);count=sum(O.read(p)['worker']==index for p in (CONTROL/'abandoned_runs').glob('*.json'))
    if failed['reason']or count>=3:return None,job['generation'],dict(phase='latched',reason=failed['reason']or'three_failed_generations')
    if time.time()<failed['retry_after_epoch']:return None,job['generation'],dict(phase='retry_backoff')
    return None,job['generation'],None

def report(execution):
    path=ROOT/'parallel_v3/REPORT_COMPLETE.json'
    unit='synrfp-preset44-v3-report-'+execution['partition_plan_sha256'][:8]
    if path.exists():
        O.tree_audit(path);props=C.props(unit)
        if props.get('ActiveState') in C.ACTIVE or props.get('MainPID','0')!='0':return dict(phase='waiting_report_service_exit')
        return dict(phase='complete')
    job=dict(id='preset44-v3-report',unit=unit,python=O.PYTHON,
        arguments=[str(HERE/'report_parallel_v3.py'),'--root',str(ROOT),'--once'],cpu_equivalents=1,memory_gib=7,cores='18',
        stage='report',sources_sha256=execution['sources_sha256'],environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        work_done=dict(kind='receipt',path=str(path),equals=dict(phase='complete')))
    return C.ensure(job,D)

def cycle():
    admitted=manifest()
    if admitted is None:return dict(phase='waiting_tested_five_shard_review')
    partition,execution=admitted
    if not old_workers_done():return dict(phase='waiting_old_single_worker_audited_yield_and_exit')
    jobs=dispatched();done={i:worker_complete(i,jobs,partition,execution)for i in range(5)}
    if all(done.values()):
        if any(C.props(j['unit']).get('ActiveState')in C.ACTIVE or C.props(j['unit']).get('MainPID','0')!='0'for j in jobs):
            return dict(phase='waiting_five_scientific_services_exit')
        state=report(execution)
        if state['phase']=='complete':
            restored=O.restore41(jobs)
            return dict(phase='complete'if restored['phase']=='complete'else'path_complete_waiting_41_restore',restore41=restored)
        return dict(phase='waiting_five_way_report',report=state)
    value=O.snapshot();ctx=O.priority_context(value);previous=CONTROL/'pressure.json'
    pressure=O.pressure_state(value,ctx['active'],O.read(previous).get('held',False)if previous.exists()else False)
    O.write(previous,dict(pressure,updated_utc=B.now()));reasons=O.admission(value,0,0)
    live=[j for j in jobs if C.props(j['unit']).get('ActiveState')in C.ACTIVE]
    if pressure['held']or reasons:
        for job in live:request_yield(job)
        return dict(phase='yielding_owned_shards_for_shared_resources',pressure=pressure,reasons=reasons)
    decisions={}
    for i in range(5):
        if done[i]:continue
        job,generation,blocked=latest(i,jobs,partition,execution)
        if blocked:decisions[str(i)]=blocked;continue
        if job is None:
            if(ROOT/'HOLD_NEW_FITS').exists():continue
            reasons=O.admission(O.snapshot(),1,7*O.GIB)
            if reasons:decisions[str(i)]=dict(phase='waiting_resources',reasons=reasons);continue
            job=job_spec(i,generation+1,execution);publish(job);jobs.append(job)
        active=C.props(job['unit']).get('ActiveState')in C.ACTIVE
        O.need(active or not(ROOT/'workers'/job['run_id']).exists(),'Retained invocation must advance generation')
        decisions[str(i)]=C.ensure(job,D,held=(ROOT/'HOLD_NEW_FITS').exists())
        if decisions[str(i)]['phase']=='resumed_or_launched':reservation_visible(job)
    return dict(phase='running_or_waiting',workers_done=done,decisions=decisions,workers=5,cpu=5,memory_gib=35,gpu_jobs=0)

def preflight():
    total=0
    for pattern in ('test_parallel_v3.py','test_report_parallel_v3.py'):
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p',pattern,'-v'],capture_output=True,text=True)
        O.need(p.returncode==0,'Five-way controls failed: '+p.stdout+p.stderr);m=re.search(r'Ran (\d+) tests',p.stderr);O.need(m,'Missing five-way test count');total+=int(m.group(1))
    value=dict(schema='task-preset44-five-shard-code-ready-v3',phase='complete',tests_passed=total,sources_sha256=P.sources())
    O.immutable(ROOT/'parallel_v3/CODE_READY.json',value)
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    if a.preflight:print(preflight());return
    if a.prepare:print(P.prepare(ROOT));return
    with(ROOT/'control/controller.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(CONTROL/'FAILED.json').exists():return
        try:
            while True:
                v={**cycle(),'updated_utc':B.now()};O.write(CONTROL/'status.json',v)
                if a.once or v['phase']=='complete':return
                time.sleep(30)
        except Exception as error:
            O.write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),scientific_artifacts_preserved=True,other_worker_mutations=0,updated_utc=B.now()));raise

if __name__=='__main__':main()
