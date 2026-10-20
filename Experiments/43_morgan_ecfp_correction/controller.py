"""Reviewed corrected-Morgan scheduling; never changes scientific producers."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path.home()/'.local/share/synrfp-runs/morgan_ecfp_correction_v1'
CONTROL=ROOT/'control'
PYTHON='/home/labhhc4/anaconda3/envs/synrfp/bin/python'
PYTHONS={stage:('/home/labhhc4/anaconda3/envs/synprop/bin/python' if stage=='mlp' else PYTHON) for stage in ('encode','clustering','mlp','report')}
ENTRY=HERE/'worker.py'
OWN=('controller.py','test_controller.py','EXECUTION.md','synrfp-morgan43-v1.service')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
Q=module('morgan43_persistence',REPO/'Experiments/39_ncs_execution/controller.py')
K=module('morgan43_frozen_resource_controller',REPO/'Experiments/42_knn_factorial/controller.py')
C,B,R=Q.C,Q.B,K.R
Q.ROOT=CONTROL;C.ROOT=CONTROL/'managed';B.ROOT=C.ROOT
D=C.Digests();RAW_SNAPSHOT=Q.resource_snapshot;GIB=1024**3
SPECIALISTS=Path.home()/'.local/share/synrfp-runs/specialist_references_v1'
T5_UNIT='synrfp-specialists-t5chem-run-002-rd12f5d4562'
LANES={'e0':('encode',0,1,6,16,'8-13'),'c0':('clustering',0,2,4,12,'14-17'),
       'c1':('clustering',1,2,4,12,'20-23'),'m0':('mlp',0,1,2,16,'24-25')}
PAUSED_GOVERNORS=('synrfp-knn-factorial-v1.service','synrfp-specialists-controller-runtime-v2.service')


def need(v,message):
    if not v:raise C.IntegrityError(message)
def read(path):return C.read(path)
def write(path,v):B.write(path,v)
def pins(catalog):
    need(isinstance(catalog,dict) and bool(catalog),'Missing provenance catalogue')
    for p,h in catalog.items():need(Path(p).is_absolute(),'Relative provenance binding');D.verify(p,h)
def immutable(path,v):
    if Path(path).exists():need(read(path)==v,'Immutable43 receipt changed: '+str(path))
    else:write(path,v)
def sources():
    paths=[HERE/n for n in OWN]+[REPO/'Experiments'/p for p in
        ('39_ncs_execution/controller.py','42_knn_factorial/controller.py','42_knn_factorial/resource85.py',
         '26_ncs_continuous/controller.py','25_followup_scheduler/budget.py')]
    return {str(p.resolve()):B.sha(p) for p in paths}
def snapshot():
    if (ROOT/'ARTIFACT_STORAGE.json').exists():storage_guard()
    return R.snapshot(raw_snapshot=RAW_SNAPSHOT,artifact_root=ROOT/'artifacts')
def context(value):return K.priority_context(value)
def admission(value,cpu,memory):return R.admission(value,cpu,memory,context(value))
B.snapshot=snapshot;B.admission=admission


def tree_audit(path,depth=0,seen=None):
    """Check native nested receipts without loading arrays or fitting libraries."""
    path=Path(path);seen=set() if seen is None else seen
    if str(path.resolve()) in seen:return
    need(depth<6,'Unexpected receipt nesting');seen.add(str(path.resolve()))
    try:v=json.loads(path.read_text())
    except (ValueError,OSError) as error:raise C.IntegrityError('Malformed nested JSON artifact: '+str(path)) from error
    if not isinstance(v,dict):
        need(depth>0,'Top-level receipt must be a JSON object');return
    for field in ('sources_sha256',):
        if v.get(field):pins(v[field])
    for field in ('outputs_sha256','artifacts_sha256','files'):
        if field not in v:continue
        catalog=v[field];need(isinstance(catalog,dict),'Malformed nested artifact catalogue')
        base=Path(v.get('attempt_directory',path.parent)) if field=='artifacts_sha256' else path.parent
        for name,h in catalog.items():
            p=base/name;D.verify(p,h)
            if p.suffix=='.json':tree_audit(p,depth+1,seen)
    for field,suffix in (('assignments_sha256','.npz'),('centers_sha256','.model.npz')):
        if field in v:D.verify(path.with_suffix(suffix),v[field])


def storage_guard():
    r=read(ROOT/'ARTIFACT_STORAGE.json');need(r.get('phase')=='complete' and r.get('existing_files_moved_or_deleted') is False,'Storage receipt boundary changed')
    need(Path(r['logical_artifacts_path']).resolve()==Path(r['physical_artifacts_path']).resolve()==(ROOT/'artifacts').resolve(),'Resolved43 artifact location changed')

def manifest():
    paths={n:ROOT/n for n in ('CORE_READY.json','CODE_READY.json','PLAN_READY.json','PILOT_READY.json','ROOT_REVIEW.json','plan.json','report_CODE_READY.json','ARTIFACT_STORAGE.json','ENVIRONMENT_READY.json')}
    ownpath=CONTROL/'CONTROLLER_CODE_READY.json'
    if not ownpath.exists() or not all(p.exists() for p in paths.values()):return None
    storage_guard()
    own=read(ownpath);science=read(paths['CODE_READY.json']);core=read(paths['CORE_READY.json']);pilot=read(paths['PILOT_READY.json']);review=read(paths['ROOT_REVIEW.json']);plan=read(paths['plan.json']);ph=B.sha(paths['plan.json'])
    need(own.get('sources_sha256')==sources(),'Controller readiness source scope changed')
    need(str(ENTRY) in science.get('sources_sha256',{}) and str(HERE/'study.py') in science['sources_sha256'],'Scientific entrypoint is not bound')
    for v in (own,science,core):need(v.get('phase')=='complete' and v.get('tests_passed',0)>0,'Untested43 source readiness');pins(v['sources_sha256'])
    need(science.get('plan_sha256')==ph and science.get('core_ready_sha256')==B.sha(paths['CORE_READY.json'])
         and science.get('expected_mlp')==1680 and science.get('expected_clustering')==1440 and science.get('feature_pools')==216,'Wrong43 scientific readiness scope')
    need(core.get('tokenizer_profile')=='synrfp-ecfp-count-v1','Wrong corrected tokenizer profile')
    prepared=read(paths['PLAN_READY.json']);need(prepared.get('phase')=='complete' and prepared.get('plan_sha256')==ph,'Plan readiness binding differs')
    need(plan.get('counts')==dict(classification=1680,clustering=1440) and len(plan['configs'])==24 and len(plan['sources'])==9
         and len(plan['tasks']['classification'])==1680 and len(plan['tasks']['clustering'])==1440,'Wrong full43 plan population')
    need(pilot.get('phase')=='complete' and pilot.get('passed') is True and pilot.get('plan_sha256')==ph
         and pilot.get('actual_encoding') is True and pilot.get('configurations')==24 and pilot.get('encoded_rows',0)>=64
         and pilot.get('actual_clustering') is True and pilot.get('center_replay_passed') is True
         and pilot.get('original_mlp_initialization_executed') is True,'Actual corrected43 CPU pilot absent')
    pins(pilot['sources_sha256']);pins(pilot['outputs_sha256'])
    need(pilot['sources_sha256']==science['sources_sha256'],'Pilot scientific source differs')
    report=read(paths['report_CODE_READY.json']);need(report.get('phase')=='complete' and report.get('tests_passed',0)>0 and str(HERE/'report.py') in report.get('sources_sha256',{}),'Untested report readiness');pins(report['sources_sha256'])
    resume_path=ROOT/'operations/REFERENCE_RESUME_READY.json'
    if not resume_path.exists():return None
    resume=read(resume_path);need(resume.get('schema')=='morgan43-reference-resume-ready-v1' and resume.get('phase')=='reviewed' and resume.get('prior_enabled') is True and resume.get('prior_active') is True,'Reference restore intent missing');pins(resume['sources_sha256'])
    environment=read(paths['ENVIRONMENT_READY.json']);need(environment.get('schema')=='morgan43-stage-environments-v1' and environment.get('phase')=='complete' and environment.get('stage_python')==PYTHONS,'Unmatched43 stage interpreter profile');pins(environment['sources_sha256'])
    need(environment.get('cpu_pilot_receipt_sha256')==B.sha(paths['PILOT_READY.json']),'Matched CPU pilot binding differs')
    original=Path(environment['original_mlp_initialization_pilot_path']);D.verify(original,environment['original_mlp_initialization_pilot_sha256']);initial=read(original)
    need(initial.get('phase')=='complete' and initial.get('plan_sha256')==ph and initial.get('original_mlp_initialization_executed') is True and initial.get('sources_sha256')==science['sources_sha256'],'Original MLP initialization environment proof differs');pins(initial['outputs_sha256'])
    required={**core['sources_sha256'],**science['sources_sha256'],**own['sources_sha256'],**report['sources_sha256'],**resume['sources_sha256'],**environment['sources_sha256']}
    need(review.get('schema')=='morgan43-root-reviewed-v1' and review.get('phase')=='reviewed'
         and review.get('plan_sha256')==ph and set(review.get('stages_allowed',[]))=={'encode','clustering','mlp'},'Wrong43 root review scope')
    for field,name in (('core_ready_sha256','CORE_READY.json'),('science_code_ready_sha256','CODE_READY.json'),('pilot_receipt_sha256','PILOT_READY.json')):
        need(review.get(field)==B.sha(paths[name]),'Root review receipt binding differs: '+field)
    need(review.get('controller_code_ready_sha256')==B.sha(ownpath),'Root controller review binding differs')
    need(review.get('report_code_ready_sha256')==B.sha(paths['report_CODE_READY.json']),'Root report review binding differs')
    need(review.get('reference_resume_ready_sha256')==B.sha(resume_path),'Root reference restore review binding differs')
    need(review.get('environment_ready_sha256')==B.sha(paths['ENVIRONMENT_READY.json']),'Root environment review binding differs')
    need(all(review.get('sources_sha256',{}).get(p)==h for p,h in required.items()),'Root review omits tested source');pins(review['sources_sha256'])
    catalog={**review['sources_sha256'],str(ownpath):B.sha(ownpath),str(resume_path):B.sha(resume_path),str(original):B.sha(original)}
    catalog.update({str(p):B.sha(p) for p in paths.values()})
    launch=SPECIALISTS/'control/launches'/(T5_UNIT+'.json')
    need(launch.exists(),'Original T5 owner launch binding missing');catalog[str(launch)]=B.sha(launch)
    v=dict(schema='morgan43-controller-execution-v1',plan_sha256=ph,sources_sha256=catalog,expected_mlp=1680,expected_clustering=1440,feature_pools=216,
           gpu=1,gpu0_reserved=True,no_42_automatic_resume=True,lanes={k:list(v) for k,v in LANES.items()},policy_fraction=.85)
    immutable(CONTROL/'execution.json',v);return v


def t5_finished():
    launch=SPECIALISTS/'control/launches'/(T5_UNIT+'.json');spec=read(launch)['spec'];need(spec['unit']==T5_UNIT and spec.get('gpu')==1,'Unexpected previous GPU owner')
    pins(spec['sources_sha256']);props=C.props(T5_UNIT);C.attest_service(spec,props)
    if props.get('ActiveState')!='inactive' or props.get('MainPID','0')!='0':return False
    ticket=SPECIALISTS/'control/tickets/t5chem/002.json'
    if not ticket.exists() or read(ticket).get('phase')=='running':return False
    r=read(ticket);ph=B.sha(SPECIALISTS/'t5chem/plan.json')
    need(r.get('phase')=='complete' and r.get('plan_sha256')==ph and r.get('completed_model_ids')==['native_seed43'],'T5 ticket002 has wrong complete model/plan')
    native=SPECIALISTS/'t5chem/models/native_seed43/COMPLETE.json'
    need(r.get('artifacts_sha256')=={str(native):B.sha(native)},'T5 ticket002 native closure differs')
    n=read(native);need(n.get('phase')=='complete' and n.get('split')=='native_seed43' and n.get('development_replay') is True,'T5 native replay incomplete')
    need(bool(n.get('artifacts_sha256')),'T5 native artifacts absent');tree_audit(ticket)
    return B.lease_available(1)


def worker_spec(lane,generation,execution):
    stage,index,workers,cpu,memory,cores=LANES[lane];run=lane+'-g'+str(generation).zfill(4)
    args=[str(ENTRY),'--root',str(ROOT),'--stage',stage,'--worker',str(index),'--workers',str(workers),'--threads',str(cpu),'--run-id',run]
    if stage=='encode':args+=['--encode-processes','4']
    job=dict(id='morgan43-'+run,unit='synrfp-morgan43-'+run+'-'+execution['plan_sha256'][:8],python=PYTHONS[stage],arguments=args,
        cpu_equivalents=cpu,memory_gib=memory,cores=cores,sources_sha256=execution['sources_sha256'],lane=lane,generation=generation,
        stage=stage,worker=index,workers=workers,run_id=run,environment=dict(CUDA_VISIBLE_DEVICES='1' if stage=='mlp' else '',
        OMP_NUM_THREADS=str(cpu),MKL_NUM_THREADS=str(cpu),OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8'),
        work_done=dict(kind='receipt',path=str(CONTROL/'closed_runs'/(run+'.json')),equals=dict(phase='complete',run_id=run)))
    if stage=='mlp':job.update(gpu=1,exclusive_units=[T5_UNIT])
    return job

def publish(job):
    path=CONTROL/'dispatch'/(job['run_id']+'.json');immutable(path,job);immutable(CONTROL/'dispatch_seals'/path.name,dict(sha256=B.sha(path)))
def dispatched():
    result=[]
    for p in sorted((CONTROL/'dispatch').glob('*.json')):
        D.verify(p,read(CONTROL/'dispatch_seals'/p.name)['sha256']);j=read(p)
        need(j.get('lane') in LANES and re.fullmatch(j['lane']+r'-g[0-9]{4,}',j.get('run_id','')) is not None,'Unknown43 job')
        pins(j['sources_sha256']);C.attest_service(j,C.props(j['unit']));result.append(j)
    return result

def task_id(t):return t['task']+'__'+t['dataset']+'__'+('f'+str(t['fold']) if t['task']=='classification' else 's'+str(t['seed']))+'__'+t['config']
def expected_paths(plan,lane):
    stage,index,workers,*_=LANES[lane]
    if stage=='encode':return {str(Path(s['feature_receipt'])) for s in plan['sources'].values()}
    values=sorted(plan['tasks']['classification' if stage=='mlp' else 'clustering'],key=lambda t:(plan['panels'][t['dataset']]['n'],t['config'],task_id(t)))
    out=[]
    for t in values[index::workers]:
        out.append(ROOT/'models'/task_id(t)/'result.json' if stage=='mlp' else ROOT/'artifacts/clustering'/t['dataset']/t['config']/(str(t['seed'])+'.json'))
    return {str(p) for p in out}

def validate_terminal(path,job,execution,plan,complete=False):
    r=read(path);kind='complete' if complete else r.get('phase');need(kind in ('complete','yielded'),'Unexpected43 terminal phase')
    need(r.get('phase')==kind and r.get('stage')==job['stage'] and r.get('worker')==job['worker'] and r.get('workers')==job['workers']
         and r.get('run_id')==job['run_id'] and r.get('plan_sha256')==execution['plan_sha256'],'Wrong43 terminal identity')
    pins(r['sources_sha256']);need(r['sources_sha256']==plan['sources_sha256'],'Terminal scientific source catalogue differs');need(all(execution['sources_sha256'].get(p)==h for p,h in r['sources_sha256'].items()),'Terminal science source not reviewed')
    expected=expected_paths(plan,job['lane']);catalog=r.get('outputs_sha256',{});need(isinstance(catalog,dict) and set(catalog)<=expected,'Wrong43 terminal output keys')
    need(r.get('expected_items')==len(expected) and r.get('audited_items')==len(catalog),'Terminal43 count mismatch')
    if kind=='complete':need(set(catalog)==expected and r.get('all_shard_items_verified') is True,'Full43 shard not verified')
    else:need(len(catalog)<len(expected) and r.get('all_shard_items_verified') is False,'Invalid43 yield closure')
    for p,h in catalog.items():D.verify(p,h);tree_audit(p)
    return kind

def terminal(job,execution,plan):
    base=ROOT/'workers'/job['run_id'];paths=[p for p in (base/'COMPLETE.json',base/'YIELDED.json') if p.exists()]
    need(len(paths)<2,'Ambiguous43 terminal receipts')
    if not paths:return None
    path=paths[0];need(read(path).get('phase')==('complete' if path.name=='COMPLETE.json' else 'yielded'),'Terminal filename phase differs');kind=validate_terminal(path,job,execution,plan)
    immutable(CONTROL/'closed_runs'/(job['run_id']+'.json'),dict(phase='complete',run_id=job['run_id'],kind=kind,outputs_sha256={str(path):B.sha(path)}))
    return kind

def lane_complete(lane,execution,plan,jobs):
    stage,index,*_=LANES[lane];path=ROOT/(stage+'_worker_'+str(index)+'_COMPLETE.json')
    if not path.exists():return False
    r=read(path);matches=[j for j in jobs if j['lane']==lane and j['run_id']==r.get('run_id')]
    need(len(matches)==1,'Full43 closure has unknown invocation');validate_terminal(path,matches[0],execution,plan,complete=True);return True

def request_yield(job):
    need(job.get('lane') in LANES and job.get('id')=='morgan43-'+job.get('run_id',''),'May signal only own43 job')
    p=C.props(job['unit']);C.attest_service(job,p)
    if p.get('ActiveState') not in C.ACTIVE:return False
    path=ROOT/'workers'/job['run_id']/'STARTED.json'
    if not path.exists():return False
    r=read(path);need(r.get('phase')=='ready' and r.get('run_id')==job['run_id'] and r.get('stage')==job['stage'],'Invalid43 handler readiness')
    if str(r.get('pid'))!=p.get('MainPID'):return False
    need(r.get('plan_sha256')==job['sources_sha256'].get(str(ROOT/'plan.json')),'Wrong yield plan')
    need(r.get('sources_sha256',{}).get(str(ENTRY))==job['sources_sha256'].get(str(ENTRY)) and all(job['sources_sha256'].get(p)==h for p,h in r.get('sources_sha256',{}).items()),'Yield handler source identity changed')
    request=CONTROL/'yield_requests'/(job['run_id']+'.json')
    if not request.exists():write(request,dict(phase='requested',run_id=job['run_id'],pid=p['MainPID'],updated_utc=B.now()))
    subprocess.run(['systemctl','--user','kill','--kill-whom=main','--signal=SIGUSR1',job['unit']],check=True);return True


def pressure_state(value,priority_active,previous=False):
    # Old42's 44/48GiB disk hysteresis would never recover after retained43 outputs.
    neutral={**value,'disk_free_bytes':100*GIB};v=R.pressure(neutral,priority_active,previous)
    fraction=value['memory_available_bytes']/value['memory_total_bytes'];low=.30 if priority_active else .15;recover=.35 if priority_active else .20
    disk=value['disk_free_bytes'];held=fraction<low or disk<42*GIB or (previous and (fraction<recover or disk<43*GIB))
    return {**v,'yield_required':held,'thaw_allowed':fraction>=recover and disk>=43*GIB,'disk_low_gib':42,'disk_recovery_gib':43}

def pressure(jobs):
    path=CONTROL/'pressure.json';prior=read(path) if path.exists() else dict(held=False)
    value=snapshot();ctx=context(value);state=pressure_state(value,ctx['active'],prior.get('held',False))
    reasons=[x for x in R.admission(value,0,0,ctx) if x!='clustering_priority_ram_launch_35pct']
    live=[j for j in jobs if C.props(j['unit']).get('ActiveState') in C.ACTIVE];chosen=[]
    if state['yield_required']:chosen=list(live)
    elif reasons:
        projected=dict(value)
        for j in sorted(live,key=lambda j:({'clustering':0,'encode':1,'mlp':2}[j['stage']],-j['worker'])):
            chosen.append(j);projected['cpu_cap_sum']-=j['cpu_equivalents'];projected['memory_cap_sum_bytes']-=j['memory_gib']*GIB
            if not [x for x in R.admission(projected,0,0,ctx) if x!='clustering_priority_ram_launch_35pct']:break
    for j in chosen:request_yield(j)
    v=dict(held=state['yield_required'] or bool(reasons),reasons=reasons,context=ctx,state=state,yielding_runs=[j['run_id'] for j in chosen],updated_utc=B.now())
    write(path,v);return v


def failed_generation(job,props):
    """Never restart an existing invocation directory; preserve and advance ID."""
    path=CONTROL/'abandoned_runs'/(job['run_id']+'.json')
    if path.exists():return read(path)
    logs=C.failure_logs(job,props,{})
    reason=C.fatal_reason(props.get('Result'),logs)
    v=dict(phase='abandoned',lane=job['lane'],run_id=job['run_id'],unit=job['unit'],reason=reason,logs=logs,
           retry_after_epoch=time.time()+30,artifacts_preserved=True,updated_utc=B.now())
    immutable(path,v);return v

def latest_job(lane,jobs,execution,plan):
    prior=[j for j in jobs if j['lane']==lane]
    if not prior:return None,0,None
    j=prior[-1];p=C.props(j['unit']);kind=terminal(j,execution,plan)
    if p.get('ActiveState') in C.ACTIVE:return j,j['generation'],None
    if kind:return None,j['generation'],None
    attempted=(ROOT/'workers'/j['run_id']).exists() or bool(p.get('InvocationID'))
    if attempted:
        failed=failed_generation(j,p);failures=[p for p in (CONTROL/'abandoned_runs').glob(lane+'-*.json')]
        if failed['reason'] or len(failures)>=3:return None,j['generation'],dict(phase='latched',reason=failed['reason'] or 'three_failed_generations')
        if time.time()<failed['retry_after_epoch']:return None,j['generation'],dict(phase='retry_backoff')
        return None,j['generation'],None
    return j,j['generation'],None

def reservation_visible(job,execution,plan):
    for _ in range(5):
        value=snapshot();units={u['unit'].removesuffix('.service'):u for u in value['units']};u=units.get(job['unit'].removesuffix('.service'))
        if u is not None:
            need(abs(u['cpu_equivalents']-job['cpu_equivalents'])<1e-8 and u['memory_max_bytes']==job['memory_gib']*GIB,'New43 cgroup reservation differs')
            C.attest_service(job,C.props(job['unit']));return True
        if C.props(job['unit']).get('ActiveState') not in C.ACTIVE:
            if job.get('stage')=='report' and report_complete(execution):return True
            if job.get('stage')!='report' and terminal(job,execution,plan):return True
        time.sleep(.1)
    raise C.IntegrityError('New43 cgroup reservation not visible before next admission')

def features_available(plan):
    for spec in plan['sources'].values():
        p=Path(spec['feature_receipt'])
        if p.exists():
            r=read(p);need(r.get('phase')=='complete' and r.get('plan_sha256')==B.sha(ROOT/'plan.json'),'Wrong43 features');tree_audit(p);return True
    return False

def report_complete(execution):
    path=ROOT/'COMPLETE.json'
    if not path.exists():return False
    r=read(path);need(r.get('schema')=='morgan43-full-correction-complete-v1' and r.get('phase')=='complete'
        and r.get('corrected_mlp_cells')==1680 and r.get('corrected_clustering_cells')==1440
        and r.get('combined_classification_cells')==5040 and r.get('combined_clustering_cells')==4320
        and r.get('all_keyed_closures_verified') is True and r.get('plan_sha256')==execution['plan_sha256']
        and r.get('tokenizer_profile')=='synrfp-ecfp-count-v1','Wrong43 full report completion scope')
    expected={str(ROOT/'report'/task/'COMPLETE.json') for task in ('classification','clustering')}
    need(set(r.get('outputs_sha256',{}))==expected,'Wrong43 report task receipt set');tree_audit(path);return True

def ensure_report(execution,plan):
    if report_complete(execution):return dict(phase='complete')
    jobs=[]
    for p in sorted((CONTROL/'report_dispatch').glob('*.json')):
        D.verify(p,read(CONTROL/'report_dispatch_seals'/p.name)['sha256']);j=read(p);pins(j['sources_sha256']);C.attest_service(j,C.props(j['unit']));jobs.append(j)
    j=jobs[-1] if jobs else None
    if j is not None:
        props=C.props(j['unit'])
        if props.get('ActiveState') not in C.ACTIVE and props.get('InvocationID'):
            failed=failed_generation(j,props);count=len(list((CONTROL/'abandoned_runs').glob('r0-*.json')))
            if failed['reason'] or count>=3:return dict(phase='latched',reason=failed['reason'] or 'three_failed_report_generations')
            if time.time()<failed['retry_after_epoch']:return dict(phase='retry_backoff')
            j=None
    if j is None:
        generation=(jobs[-1]['generation']+1 if jobs else 1);run='r0-g'+str(generation).zfill(4)
        j=dict(id='morgan43-'+run,unit='synrfp-morgan43-'+run+'-'+execution['plan_sha256'][:8],python=PYTHON,
            arguments=[str(HERE/'report.py'),'--root',str(ROOT),'--once'],cpu_equivalents=2,memory_gib=8,cores='20-21',
            sources_sha256=execution['sources_sha256'],stage='report',lane='r0',run_id=run,generation=generation,
            environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),
            work_done=dict(kind='receipt',path=str(ROOT/'COMPLETE.json'),equals=dict(phase='complete')))
        path=CONTROL/'report_dispatch'/(run+'.json');immutable(path,j);immutable(CONTROL/'report_dispatch_seals'/path.name,dict(sha256=B.sha(path)))
    v=C.ensure(j,D)
    if v['phase']=='resumed_or_launched':reservation_visible(j,execution,plan)
    return v


def attest_reference_governor(ready,props):
    unit=ready['unit'];need(unit==PAUSED_GOVERNORS[1],'Wrong reference governor restore scope')
    D.verify(ready['unit_file'],ready['installed_unit_sha256']);pins(ready['sources_sha256'])
    need(props.get('LoadState')=='loaded','Original reference governor is not loaded')
    text=Path(ready['unit_file']).read_text();entries=[x.split('=',1)[1] for x in text.splitlines() if x.startswith('ExecStart=')]
    need(len(entries)==1,'Ambiguous original reference governor command')
    match=re.search(r'argv\[\]=(.*?) ; ignore_errors=',props.get('ExecStart',''));actual=shlex.split(match.group(1)) if match else []
    expected=shlex.split(entries[0]);need(actual==expected,'Reference governor ExecStart changed')
    need(props.get('CPUQuotaPerSecUSec') in ('100ms','100000us','0.1s') and props.get('MemoryMax')==str(256*1024**2)
         and props.get('KillMode')=='control-group' and props.get('WorkingDirectory')==str(REPO),'Reference governor caps/context changed')
    env=dict(x.split('=',1) for x in shlex.split(props.get('Environment','')))
    need(env.get('CUDA_VISIBLE_DEVICES')=='','Reference governor CUDA environment changed')

def resume_references(execution,mlp_done,jobs):
    ready=read(ROOT/'operations/REFERENCE_RESUME_READY.json');unit=ready['unit'];props=C.props(unit)
    if not mlp_done:return dict(phase='waiting_all_1680_mlp')
    mlp=[j for j in jobs if j['stage']=='mlp']
    if any(C.props(j['unit']).get('ActiveState') in C.ACTIVE or C.props(j['unit']).get('MainPID','0')!='0' for j in mlp):return dict(phase='waiting_mlp_unit_exit')
    attest_reference_governor(ready,props)
    marker=CONTROL/'REFERENCE_RESUMED.json';intent=CONTROL/'REFERENCE_RESUME_INTENT.json'
    if marker.exists():
        r=read(marker);need(r.get('phase')=='complete' and r.get('execution_sha256')==B.sha(CONTROL/'execution.json'),'Reference resumed proof changed');pins(r['outputs_sha256']);return dict(phase='complete')
    if props.get('ActiveState') in C.ACTIVE:
        need(intent.exists(),'Unexpected reference restart before durable intent')
    else:
        if not B.lease_available(1):return dict(phase='waiting_gpu1_lease_for_reference_restore')
        reasons=admission(snapshot(),.1,256*1024**2)
        if reasons:return dict(phase='waiting_resources',reasons=reasons)
        expected=dict(phase='ready',execution_sha256=B.sha(CONTROL/'execution.json'),unit=unit,
            outputs_sha256={str(ROOT/'operations/REFERENCE_RESUME_READY.json'):B.sha(ROOT/'operations/REFERENCE_RESUME_READY.json'),
                            str(ROOT/'mlp_worker_0_COMPLETE.json'):B.sha(ROOT/'mlp_worker_0_COMPLETE.json')})
        immutable(intent,expected)
        subprocess.run(['systemctl','--user','enable','--now',unit],check=True)
    for _ in range(5):
        props=C.props(unit);attest_reference_governor(ready,props)
        value=snapshot();matches=[u for u in value['units'] if u['unit'].removesuffix('.service')==unit.removesuffix('.service')]
        if props.get('ActiveState') in C.ACTIVE and len(matches)==1:
            u=matches[0];need(abs(u['cpu_equivalents']-.1)<1e-8 and u['memory_max_bytes']==256*1024**2,'Reference governor reservation changed')
            immutable(marker,dict(phase='complete',execution_sha256=B.sha(CONTROL/'execution.json'),unit=unit,outputs_sha256={str(intent):B.sha(intent),**read(intent)['outputs_sha256']}));return dict(phase='complete')
        time.sleep(.1)
    raise C.IntegrityError('Restored35 governor cgroup not visible')


def yield_for_mlp(jobs,value):
    projected=dict(value);chosen=[]
    live=[j for j in jobs if j['stage'] in ('clustering','encode') and C.props(j['unit']).get('ActiveState') in C.ACTIVE]
    for j in sorted(live,key=lambda j:(0 if j['stage']=='clustering' else 1,-j['worker'])):
        chosen.append(j);projected['cpu_cap_sum']-=j['cpu_equivalents'];projected['memory_cap_sum_bytes']-=j['memory_gib']*GIB
        if not admission(projected,2,16*GIB):
            for candidate in chosen:request_yield(candidate)
            return [x['run_id'] for x in chosen]
    return []


def cycle():
    execution=manifest()
    if execution is None:return dict(phase='waiting_tested_pilot_and_root_review')
    pins(execution['sources_sha256'])
    need(C.props(PAUSED_GOVERNORS[0]).get('ActiveState') not in C.ACTIVE,'Deferred42 governor unexpectedly active')
    plan=read(ROOT/'plan.json');jobs=dispatched();done={lane:lane_complete(lane,execution,plan,jobs) for lane in LANES}
    refs=resume_references(execution,done['m0'],jobs)
    if not done['m0']:need(C.props(PAUSED_GOVERNORS[1]).get('ActiveState') not in C.ACTIVE,'Deferred35 governor unexpectedly active')
    if all(done.values()):
        if any(C.props(j['unit']).get('ActiveState') in C.ACTIVE for j in jobs):return dict(phase='scientific_workers_finalizing',done=done)
        proof={str(ROOT/(s+'_worker_'+str(i)+'_COMPLETE.json')):B.sha(ROOT/(s+'_worker_'+str(i)+'_COMPLETE.json')) for s,i,*_ in LANES.values()}
        immutable(CONTROL/'WORKERS_COMPLETE.json',dict(phase='complete',plan_sha256=execution['plan_sha256'],expected_mlp=1680,expected_clustering=1440,feature_pools=216,outputs_sha256=proof))
        report=ensure_report(execution,plan)
        closed=report_complete(execution)
        return dict(phase=('complete' if refs.get('phase')=='complete' else 'report_complete_waiting_reference_restore') if closed else 'workers_complete_waiting_report',done=done,report=report,reference_resume=refs)
    reserve=pressure(jobs);decisions={}
    for lane in ('m0','e0','c0','c1'):
        if done[lane]:continue
        j,generation,blocked=latest_job(lane,jobs,execution,plan)
        if blocked:decisions[lane]=blocked;continue
        if j is None:
            if lane!='m0' and decisions.get('m0',{}).get('yield_requested'):continue
            if reserve['held'] or (ROOT/'HOLD_NEW_FITS').exists():continue
            if lane!='e0' and not features_available(plan):continue
            if lane=='m0' and not t5_finished():decisions[lane]=dict(phase='waiting_original_t5_native_completion_and_gpu1_lease');continue
            candidate=worker_spec(lane,generation+1,execution)
            value=snapshot()
            if admission(value,candidate['cpu_equivalents'],candidate['memory_gib']*GIB):
                if lane=='m0':decisions[lane]=dict(phase='waiting_cpu_workers_cooperatively_yield',yield_requested=yield_for_mlp(jobs,value))
                continue
            publish(candidate);jobs.append(candidate);j=candidate
        p=C.props(j['unit']);already=p.get('ActiveState') in C.ACTIVE
        # No same-generation retry may cross an existing invocation directory.
        need(already or not (ROOT/'workers'/j['run_id']).exists(),'Existing43 invocation cannot restart')
        gpu_wait=j['stage']=='mlp' and not already and not t5_finished()
        decisions[lane]=C.ensure(j,D,held=reserve['held'] or gpu_wait or (ROOT/'HOLD_NEW_FITS').exists())
        if decisions[lane]['phase']=='resumed_or_launched':reservation_visible(j,execution,plan)
    return dict(phase='running_or_waiting',done=done,decisions=decisions,resource_pressure=reserve,expected_mlp=1680,expected_clustering=1440,feature_pools=216,gpu=1,no_42_automatic_resume=True,reference_resume=refs)


def preflight():
    p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_controller.py','-v'],capture_output=True,text=True)
    need(p.returncode==0,'43controller tests failed: '+p.stdout+p.stderr);m=re.search(r'Ran (\d+) tests',p.stderr);need(m is not None,'Controller test count absent')
    v=dict(schema='morgan43-controller-code-ready-v1',phase='complete',tests_passed=int(m.group(1)),sources_sha256=sources(),lanes={k:list(v) for k,v in LANES.items()},controller_cpu=.1,controller_memory_bytes=256*1024**2,policy_fraction=.85,production_launched=False,ready_utc=B.now())
    write(CONTROL/'CONTROLLER_CODE_READY.json',v);return v

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args();CONTROL.mkdir(parents=True,exist_ok=True)
    if a.preflight:print(preflight());return
    with (CONTROL/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (CONTROL/'FAILED.json').exists():return
        try:
            while True:
                value=dict(cycle(),updated_utc=B.now());write(CONTROL/'status.json',value)
                if a.once or value['phase']=='complete':return
                time.sleep(30)
        except Exception as error:
            requested=[];cleanup_error=None
            try:
                for path in sorted((CONTROL/'dispatch').glob('*.json')):
                    D.verify(path,read(CONTROL/'dispatch_seals'/path.name)['sha256']);j=read(path)
                    if request_yield(j):requested.append(j['run_id'])
            except Exception as cleanup:cleanup_error=repr(cleanup)
            write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),yield_requested=requested,cleanup_error=cleanup_error,scientific_artifacts_preserved=True,other_task_mutations=0,updated_utc=B.now()));raise

if __name__=='__main__':main()
