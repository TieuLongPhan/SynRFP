"""Persistent full-row kNNfactorial with source-specific85% shared admission."""
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
REPO=HERE.parents[1]
ROOT=Path.home()/'.local/share/synrfp-runs/knn_factorial_full_v1'
CONTROL=ROOT/'control'
PYTHON='/home/labhhc4/anaconda3/envs/synrfp/bin/python'
ENTRY=HERE/'worker.py'
OWN=('controller.py','test_controller.py','RESOURCE_PROTOCOL.md','synrfp-knn-factorial-v1.service','resource85.py','test_resource85.py')
s=importlib.util.spec_from_file_location('knn42_frozen_persistence',REPO/'Experiments/39_ncs_execution/controller.py')
Q=importlib.util.module_from_spec(s);s.loader.exec_module(Q)
C,B=Q.C,Q.B
Q.ROOT=CONTROL;C.ROOT=CONTROL/'managed';B.ROOT=C.ROOT
s=importlib.util.spec_from_file_location('knn42_resource85',HERE/'resource85.py')
R=importlib.util.module_from_spec(s);s.loader.exec_module(R)
D=C.Digests();RAW_SNAPSHOT=Q.resource_snapshot
GIB=1024**3


def need(value,message):
    if not value:raise C.IntegrityError(message)
def read(path):return C.read(path)
def write(path,value):B.write(path,value)
def pins(values):
    need(isinstance(values,dict) and bool(values),'Missing provenance catalogue')
    for p,h in values.items():need(Path(p).is_absolute(),'Relative provenance path');D.verify(p,h)
def immutable(path,value):
    if Path(path).exists():need(read(path)==value,'Immutable42record changed: '+str(path))
    else:write(path,value)
def sources():
    paths=[HERE/n for n in OWN]+[REPO/'Experiments/39_ncs_execution/controller.py',REPO/'Experiments/26_ncs_continuous/controller.py',REPO/'Experiments/25_followup_scheduler/budget.py']
    return {str(p.resolve()):B.sha(p) for p in paths}

def snapshot():return R.snapshot(raw_snapshot=RAW_SNAPSHOT,artifact_root=ROOT)

def priority_terminal(root,value):
    state=root/'control_priority_v2';complete=state/'COMPLETE.json';quiescent=state/'QUIESCENT.json'
    path=complete if complete.exists() else quiescent
    if not path.exists():return False
    v=read(path);execution=state/'execution.json'
    need(v.get('execution_sha256')==B.sha(execution),'41terminal execution binding changed')
    if path==complete:
        need(v.get('phase')=='complete' and v.get('counts')==dict(mapped=9,native=10,comparison=10)
             and v.get('endpoints')==12 and v.get('all_keyed_stage_closures_verified') is True,'Invalid41complete scope')
        catalog=v.get('outputs_sha256')
    else:
        need(v.get('phase')=='quiescent','Invalid41terminal quiescence')
        status=Path(v.get('status_path',''));need(status.resolve()==(state/'status.json').resolve(),'Wrong41quiescent status path');D.verify(status,v['status_sha256']);q=read(status)
        need(q.get('execution_sha256')==B.sha(execution) and q.get('counts',{}).get('mapped')==9
             and q.get('endpoints')==12 and q.get('fully_idle') is True and q.get('safe_to_yield_cpu') is True
             and q.get('active_jobs')==[] and q.get('pending_runnable')==[]
             and bool(q.get('resource_refusals') or q.get('scientific_latches')),'41quiescence is not a terminal scoped release')
        catalog=v.get('verified_receipts_sha256');need(catalog==q.get('verified_receipts_sha256'),'41quiescent marker catalogue changed')
        need(C.gate(dict(kind='receipt',path=str(path),equals=dict(phase='quiescent')),D),'41quiescence output closure changed')
    pins(catalog)
    for marker in catalog:
        need(Path(marker).resolve().parent==(root/'control/verified').resolve(),'41terminal marker is outside original namespace')
        need(C.gate(dict(kind='receipt',path=marker,equals=dict(phase='complete')),D),'41terminal underlying artifact changed')
    p=C.props('synrfp-syncluster41-priority-v2.service')
    if p.get('ActiveState')!='inactive' or p.get('MainPID','0')!='0' or p.get('Result')!='success':return False
    exempt={'synrfp-syncluster41-input-repair-v2.service','synrfp-syncluster41-priority-v2.service','synrfp-syncluster41-cpu-priority-v2.service'}
    return not any(u['unit'].startswith('synrfp-syncluster41-') and u['unit'] not in exempt for u in value.get('units',[]))

def priority_context(value):
    root=R.PRIORITY_ROOT
    if priority_terminal(root,value):return dict(active=False,pending_cpu=0,pending_memory_bytes=0)
    context=dict(active=True,pending_cpu=2,pending_memory_bytes=24*GIB)
    active={u['unit'].removesuffix('.service') for u in value.get('units',[])}
    pending={}
    for path in (root/'control/dispatch').glob('*.json'):
        j=read(path);stage=j.get('operation',{}).get('stage')
        if stage not in ('native','comparison') or j.get('unit','').removesuffix('.service') in active:continue
        gate=j.get('work_done',{});marker=Path(gate.get('path','/nonexistent'))
        if marker.is_file() or (root/'control/managed/latches'/(j.get('id','')+'.json')).exists():continue
        lane=j.get('lane',stage);pending[lane]=max(pending.get(lane,0),int(float(j.get('memory_gib',8))*GIB))
    if pending:context['pending_memory_bytes']=max(context['pending_memory_bytes'],sum(pending.values()))
    return context

def admission(value,cpu,memory_bytes):return R.admission(value,cpu,memory_bytes,priority_context(value))
B.snapshot=snapshot;B.admission=admission


def manifest():
    required=[CONTROL/'CONTROLLER_CODE_READY.json',ROOT/'CODE_READY.json',ROOT/'PLAN_READY.json',ROOT/'plan.json',ROOT/'PILOT_READY.json',ROOT/'ROOT_REVIEW.json']
    if not all(p.exists() for p in required):return None
    own=read(required[0]);science=read(required[1]);pilot=read(required[4]);review=read(required[5]);ph=B.sha(required[3])
    need(own.get('phase')=='complete' and own.get('tests_passed',0)>0 and science.get('phase')=='complete' and science.get('tests_passed',0)>0,'Tested42code readiness absent')
    pins(own['sources_sha256']);pins(science['sources_sha256'])
    need(science.get('plan_sha256')==ph and science.get('expected_models')==5040,'Scientific42readiness belongs to another plan')
    need(str(ENTRY) in science['sources_sha256'] and str(HERE/'study.py') in science['sources_sha256'],'Scientific42entrypoint binding absent')
    prepared=read(required[2]);need(prepared.get('phase')=='complete' and prepared.get('expected_models')==5040 and prepared.get('panels')==14 and prepared.get('configurations')==72 and prepared.get('folds')==5 and prepared.get('plan_sha256')==ph,'Prepared72x14x5grid binding differs')
    need(pilot.get('phase')=='complete' and pilot.get('passed') is True and pilot.get('actual_knn_execution') is True
         and pilot.get('prediction_replay_passed') is True and pilot.get('plan_sha256')==ph
         and type(pilot.get('peak_rss_bytes')) is int and 0<pilot['peak_rss_bytes']<16*GIB
         and type(pilot.get('full_tpl_memory_forecast_bytes')) is int and 0<pilot['full_tpl_memory_forecast_bytes']<16*GIB,'Actual6thread16GiB kNNpilot has not passed')
    pins(pilot['sources_sha256']);pins(pilot['outputs_sha256'])
    need(pilot['sources_sha256'].get(str(ENTRY))==science['sources_sha256'][str(ENTRY)],'Pilot worker source differs')
    needed={**own['sources_sha256'],**science['sources_sha256']}
    need(review.get('schema')=='knn42-root-reviewed-v1' and review.get('full_scope_approved') is True
         and review.get('plan_sha256')==ph and review.get('worker_code_ready_sha256')==B.sha(required[1])
         and review.get('pilot_receipt_sha256')==B.sha(required[4])
         and all(review.get('sources_sha256',{}).get(p)==h for p,h in needed.items()),'Root42review does not bind exact plan/code/pilot')
    pins(review['sources_sha256']);plan=read(required[3]);tasks=plan.get('tasks')
    need(isinstance(tasks,list) and len(tasks)==5040,'Full72x14x5 scope must contain5040tasks')
    catalog=dict(review['sources_sha256'])
    for p in required:catalog[str(p.resolve())]=B.sha(p)
    value=dict(schema='knn42-controller-execution-v1',sources_sha256=catalog,plan_sha256=ph,expected_cells=5040,workers=2,worker_cpu=6,worker_memory_gib=16,policy_fraction=.85,gpu_jobs=0)
    immutable(CONTROL/'execution.json',value);return value

def worker_spec(index,generation,execution):
    run='w'+str(index)+'-g'+str(generation).zfill(4)
    return dict(id='knn42-'+run,unit='synrfp-knn42-'+run+'-'+execution['plan_sha256'][:8],python=PYTHON,
        arguments=[str(ENTRY),'--root',str(ROOT),'--worker',str(index),'--workers','2','--threads','6','--run-id',run],
        cpu_equivalents=6,memory_gib=16,cores='8-13' if index==0 else '20-25',sources_sha256=execution['sources_sha256'],
        environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6'),
        work_done=dict(kind='receipt',path=str(CONTROL/'closed_runs'/(run+'.json')),equals=dict(phase='complete',run_id=run)),worker=index,run_id=run,generation=generation)

def publish(job):
    p=CONTROL/'dispatch'/(job['run_id']+'.json')
    if p.exists():need(read(p)==job,'Worker generation identity changed')
    else:write(p,job)
    seal=CONTROL/'dispatch_seals'/p.name
    immutable(seal,dict(sha256=B.sha(p)))

def dispatched():
    jobs=[]
    for p in sorted((CONTROL/'dispatch').glob('*.json')):
        D.verify(p,read(CONTROL/'dispatch_seals'/p.name)['sha256']);j=read(p)
        need(re.fullmatch(r'w[01]-g[0-9]{4,}',j.get('run_id','')) is not None and j.get('worker') in (0,1),'Unknown42worker generation')
        pins(j['sources_sha256']);C.attest_service(j,C.props(j['unit']));jobs.append(j)
    return jobs

def terminal(job,execution):
    base=ROOT/'workers'/job['run_id'];complete=base/'COMPLETE.json';yielded=base/'YIELDED.json'
    need(not (complete.exists() and yielded.exists()),'Ambiguous42terminal receipt')
    p=complete if complete.exists() else yielded
    if not p.exists():return None
    r=read(p);expected='complete' if p==complete else 'yielded'
    need(r.get('phase')==expected and r.get('run_id')==job['run_id'] and r.get('worker')==job['worker']
         and r.get('plan_sha256')==execution['plan_sha256'],'Wrong42terminal run/plan binding')
    pins(r['sources_sha256']);need(r['sources_sha256'].get(str(ENTRY))==execution['sources_sha256'][str(ENTRY)],'Terminal42worker source changed')
    outputs=r.get('outputs_sha256',{})
    need(isinstance(outputs,dict),'Terminal42output catalogue absent')
    for path,h in outputs.items():D.verify(path,h)
    marker=CONTROL/'closed_runs'/(job['run_id']+'.json')
    if marker.exists():need(C.gate(dict(kind='receipt',path=str(marker),equals=dict(phase='complete')),D),'42closed-run artifact changed')
    else:write(marker,dict(phase='complete',run_id=job['run_id'],kind=expected,outputs_sha256={**outputs,**r['sources_sha256'],str(p):B.sha(p)}))
    return expected

def worker_complete(index,execution):
    path=ROOT/('worker_'+str(index)+'_COMPLETE.json')
    if not path.exists():return False
    r=read(path)
    need(r.get('phase')=='complete' and r.get('worker')==index and r.get('workers')==2 and r.get('plan_sha256')==execution['plan_sha256']
         and r.get('audited_cells')==2520 and r.get('expected_cells')==2520 and r.get('all_shard_cells_audited') is True,'Full2520-cell shard completion differs')
    pins(r['sources_sha256']);need(r['sources_sha256'].get(str(ENTRY))==execution['sources_sha256'][str(ENTRY)],'Full-shard worker source differs')
    need(C.gate(dict(kind='receipt',path=str(path),equals=dict(phase='complete')),D),'Full-shard artifact closure failed')
    return True

def request_yield(job):
    need(re.fullmatch(r'knn42-w[01]-g[0-9]{4,}',job.get('id','')) is not None and job.get('worker') in (0,1),'May only yield own42worker')
    p=C.props(job['unit']);C.attest_service(job,p)
    if p.get('ActiveState') not in C.ACTIVE:return False
    ready=ROOT/'workers'/job['run_id']/'STARTED.json'
    if not ready.exists():return False
    r=read(ready)
    need(r.get('phase')=='ready' and r.get('handler_installed') is True and r.get('run_id')==job['run_id'] and r.get('worker')==job['worker'],'Worker yield-handler readiness invalid')
    if str(r.get('pid'))!=p.get('MainPID'):return False # Previous crashed invocation.
    path=CONTROL/'yield_requests'/(job['run_id']+'.json')
    if not path.exists():write(path,dict(phase='requested',unit=job['unit'],run_id=job['run_id'],pid=p['MainPID'],requested_utc=B.now()))
    # Idempotent resend closes a crash window between durable request and signal.
    subprocess.run(['systemctl','--user','kill','--kill-whom=main','--signal=SIGUSR1',job['unit']],check=True)
    return True

def pressure(jobs):
    path=CONTROL/'pressure.json';old=read(path) if path.exists() else dict(held=False,frozen_units=[])
    v=snapshot();context=priority_context(v);state=R.pressure(v,context['active'],old.get('held',False));need(not old.get('frozen_units'),'Unexpected frozen42unit needs preserved operational review')
    live=[j for j in jobs if C.props(j['unit']).get('ActiveState') in C.ACTIVE]
    reasons=[x for x in R.admission(v,0,0,context) if x!='clustering_priority_ram_launch_35pct']
    chosen=list(live) if state['yield_required'] else []
    if reasons and not chosen:
        projected=dict(v)
        for j in sorted(live,key=lambda x:x['worker'],reverse=True):
            chosen.append(j);projected['cpu_cap_sum']-=j['cpu_equivalents'];projected['memory_cap_sum_bytes']-=j['memory_gib']*GIB
            if not [x for x in R.admission(projected,0,0,context) if x!='clustering_priority_ram_launch_35pct']:break
    for j in chosen:
        request_yield(j)
    value=dict(held=state['yield_required'] or bool(reasons),frozen_units=[],yielding_runs=[j['run_id'] for j in chosen],reasons=reasons,context=context,pressure=state,updated_utc=B.now())
    write(path,value);return value

def ensure_report(execution):
    ready=ROOT/'report_CODE_READY.json';review=ROOT/'REPORT_ROOT_REVIEW.json'
    if not ready.exists() or not review.exists():return dict(phase='waiting_report_review')
    r=read(ready);v=read(review);pins(r['sources_sha256']);need(r.get('phase')=='complete' and r.get('tests_passed',0)>0,'Report tests not ready')
    need(v.get('phase')=='complete' and v.get('report_code_ready_sha256')==B.sha(ready) and v.get('plan_sha256')==execution['plan_sha256']
         and v.get('sources_sha256')==r['sources_sha256'],'Report root review differs')
    own={**execution['sources_sha256'],**r['sources_sha256'],str(ready):B.sha(ready),str(review):B.sha(review)}
    j=dict(id='knn42-final-report',unit='synrfp-knn42-final-report-'+execution['plan_sha256'][:8],python=PYTHON,
        arguments=[str(HERE/'report.py'),'--root',str(ROOT),'--once'],cpu_equivalents=2,memory_gib=8,cores='20-21',sources_sha256=own,
        environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),
        work_done=dict(kind='receipt',path=str(ROOT/'COMPLETE.json'),equals=dict(phase='complete')))
    return C.ensure(j,D)

def reservation_visible(job,execution):
    for attempt in range(5):
        value=snapshot();units={u['unit'].removesuffix('.service'):u for u in value['units']};unit=job['unit'].removesuffix('.service')
        if unit in units:
            r=units[unit];need(abs(r['cpu_equivalents']-job['cpu_equivalents'])<1e-8 and r['memory_max_bytes']==job['memory_gib']*GIB,'42new worker reservation has wrong finite caps')
            C.attest_service(job,C.props(job['unit']));return True
        p=C.props(job['unit'])
        if p.get('ActiveState') not in C.ACTIVE and terminal(job,execution):return True
        time.sleep(.1)
    raise C.IntegrityError('42new worker reservation is not visible before next admission')

def cycle():
    execution=manifest()
    if execution is None:return dict(phase='waiting_tested_worker_pilot_and_root_review')
    jobs=dispatched();closed={j['run_id']:terminal(j,execution) for j in jobs}
    done={i:worker_complete(i,execution) for i in (0,1)}
    if all(done.values()):
        if any(C.props(j['unit']).get('ActiveState') in C.ACTIVE for j in jobs):return dict(phase='workers_finalizing_before_report')
        proof={str(ROOT/('worker_'+str(i)+'_COMPLETE.json')):B.sha(ROOT/('worker_'+str(i)+'_COMPLETE.json')) for i in (0,1)}
        if not (CONTROL/'WORKERS_COMPLETE.json').exists():write(CONTROL/'WORKERS_COMPLETE.json',dict(phase='complete',plan_sha256=execution['plan_sha256'],cells=5040,outputs_sha256=proof,finished_utc=B.now()))
        report=ensure_report(execution)
        if (ROOT/'COMPLETE.json').exists():
            need(C.gate(dict(kind='receipt',path=str(ROOT/'COMPLETE.json'),equals=dict(phase='complete',plan_sha256=execution['plan_sha256'],completed=5040,all_keyed_cells_verified=True,metric_and_prediction_replay=True)),D),'Canonical42report closure changed')
            return dict(phase='complete',cells=5040,report=report)
        return dict(phase='workers_complete_waiting_report',report=report)
    reserve=pressure(jobs);decisions={}
    for index in (0,1):
        if done[index]:continue
        prior=[j for j in jobs if j['worker']==index];j=prior[-1] if prior else None
        if j and closed[j['run_id']]:
            if C.props(j['unit']).get('ActiveState') in C.ACTIVE:continue
            j=None
        if j is None:
            if reserve['held'] or reserve['frozen_units']:continue
            candidate=worker_spec(index,(prior[-1]['generation']+1 if prior else 1),execution)
            if admission(snapshot(),6,16*GIB):continue
            publish(candidate);jobs.append(candidate);j=candidate
        state=C.props(j['unit']);request=CONTROL/'yield_requests'/(j['run_id']+'.json')
        requested_current=request.exists() and str(read(request).get('pid'))==state.get('MainPID') and state.get('ActiveState') in C.ACTIVE
        held=reserve['held'] or requested_current
        decisions[j['run_id']]=C.ensure(j,D,held=held)
        # A newly launched unit must have an actual finite accounted cgroup
        # before considering the secondworker; freshsnapshot failure blocks.
        if decisions[j['run_id']]['phase']=='resumed_or_launched':reservation_visible(j,execution)
    return dict(phase='running_or_yielding',expected_cells=5040,worker_done=done,decisions=decisions,resource_pressure=reserve,gpu_jobs=0,policy_fraction=.85)

def preflight():
    total=0
    for pattern in ('test_controller.py','test_resource85.py'):
        r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p',pattern,'-v'],capture_output=True,text=True)
        need(r.returncode==0,'42controller/resource tests failed: '+r.stdout+r.stderr);m=re.search(r'Ran (\d+) tests',r.stderr);need(m is not None,'Test count absent');total+=int(m.group(1))
    value=dict(schema='knn42-controller-code-ready-v1',phase='complete',tests_passed=total,sources_sha256=sources(),controller_cpu=.1,controller_memory_bytes=256*1024**2,workers=2,worker_cpu=6,worker_memory_gib=16,policy_fraction=.85,gpu_jobs=0,created_utc=B.now())
    immutable(CONTROL/'CONTROLLER_CODE_READY.json',value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--once',action='store_true');args=p.parse_args();CONTROL.mkdir(parents=True,exist_ok=True)
    if args.preflight:print(preflight());return
    with (CONTROL/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (CONTROL/'FAILED.json').exists():return
        try:
            while True:
                value=dict(cycle(),updated_utc=B.now());write(CONTROL/'status.json',value)
                if args.once or value['phase']=='complete':return
                time.sleep(30)
        except Exception as error:
            # Cooperative yield only owned42, preserve workers of35/40/41.
            requested=[]
            try:
                for j in dispatched():
                    if request_yield(j):requested.append(j['run_id'])
            except Exception as cleanup:cleanup_error=repr(cleanup)
            else:cleanup_error=None
            write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),yield_requested=requested,cleanup_error=cleanup_error,scientific_artifacts_preserved=True,other_task_mutations=0,updated_utc=B.now()));raise

if __name__=='__main__':main()
