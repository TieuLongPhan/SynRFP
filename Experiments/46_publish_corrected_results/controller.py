"""Small persistent watcher; each completed component publishes independently."""
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
ROOT=Path.home()/'.local/share/synrfp-runs/corrected_results_publication_v1'
CONTROL=ROOT/'control'
PYTHON='/home/labhhc4/anaconda3/envs/synrfp/bin/python'
GIB=1024**3


def module(name,path):
    q=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(q);sys.modules[name]=m;q.loader.exec_module(m);return m


P=module('publication46_metadata_publisher',HERE/'publisher.py')
Q=module('publication46_persistence',REPO/'Experiments/39_ncs_execution/controller.py')
R=module('publication46_resource85',REPO/'Experiments/42_knn_factorial/resource85.py')
PRIORITY=module('publication46_priority_boundary',REPO/'Experiments/42_knn_factorial/controller.py')
PAUSE=module('publication46_verified_pause',REPO/'Experiments/43_morgan_ecfp_correction/priority_policy_v2.py')
C,B=Q.C,Q.B;Q.ROOT=CONTROL;C.ROOT=CONTROL/'managed';B.ROOT=C.ROOT
D=C.Digests();RAW_SNAPSHOT=Q.resource_snapshot


def sources():
    paths=[REPO/'Experiments'/n for n in ('39_ncs_execution/controller.py','26_ncs_continuous/controller.py',
      '25_followup_scheduler/budget.py','42_knn_factorial/controller.py','42_knn_factorial/resource85.py')]
    return {**P.source_pins(),**PAUSE.sources(),**{str(p.resolve()):P.sha(p)for p in paths}}


def need(v,message):
    if not v:raise C.IntegrityError(message)


def snapshot():return R.snapshot(raw_snapshot=RAW_SNAPSHOT,artifact_root=ROOT)

def admission(v,cpu,memory):
    context=dict(active=False,pending_cpu=0,pending_memory_bytes=0) if PAUSE.paused41(v) else PRIORITY.priority_context(v)
    return R.admission(v,cpu,memory,context)

B.snapshot=snapshot;B.admission=admission


def review():
    paths=[ROOT/n for n in ('ROOT_REVIEW.json','CODE_READY.json','config.json')]
    if not all(p.exists()for p in paths):return None
    r,ready=P.read(paths[0]),P.read(paths[1]);catalog=sources()
    need(r.get('schema')=='publication46-root-reviewed-v1'and r.get('phase')=='reviewed'
      and r.get('config_sha256')==P.sha(paths[2])and r.get('code_ready_sha256')==P.sha(paths[1])
      and set(r.get('components',[]))=={'path','morgan'}and r.get('sources_sha256')==catalog
      and ready.get('phase')=='complete'and ready.get('tests_passed',0)>0 and ready.get('sources_sha256')==catalog,'Publisher root review/source readiness changed')
    for p,h in catalog.items():D.verify(p,h)
    return r


def spec(component,r):
    need(component in ('path','morgan'),'Unknown publication component')
    catalog={**r['sources_sha256'],**{str((ROOT/n).resolve()):P.sha(ROOT/n)for n in ('ROOT_REVIEW.json','CODE_READY.json','config.json')}}
    return dict(id='publication46-'+component,unit='synrfp-publish46-'+component+'-'+P.sha(ROOT/'config.json')[:8],python=PYTHON,
      arguments=[str(HERE/'publisher.py'),'--component',component],cpu_equivalents=2,memory_gib=1,cores='18-19',sources_sha256=catalog,
      environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),
      work_done=dict(kind='receipt',path=str(ROOT/component/'PUBLISHED.json'),equals=dict(phase='complete',component=component)))


def completed(component):
    path=ROOT/component/'PUBLISHED.json'
    if not path.exists():return False
    r=P.read(path);need(r.get('phase')=='complete'and r.get('component')==component and r.get('branch')==P.BRANCH
      and r.get('weights_published')is False and r.get('other_component_waited_for')is False and len(r.get('commit',''))==40,'Invalid independent publication proof')
    return True


def cycle():
    r=review()
    if r is None:return dict(phase='waiting_tested_source_policy_and_root_review')
    done={c:completed(c)for c in ('path','morgan')};states={}
    if all(done.values()):
        paths={str((ROOT/c/'PUBLISHED.json').resolve()):P.sha(ROOT/c/'PUBLISHED.json')for c in done}
        result=dict(phase='complete',independent_components=done,outputs_sha256=paths,weights_published=False,finished_utc=P.now())
        if (ROOT/'COMPLETE.json').exists():need(P.read(ROOT/'COMPLETE.json')['outputs_sha256']==paths,'Publication proofs changed')
        else:P.write(ROOT/'COMPLETE.json',result)
        return result
    # One Git transaction at a time; readiness and completion are independent.
    # A latched/retrying component must not prevent the other from publishing.
    jobs={c:spec(c,r)for c in done}
    for c,j in jobs.items():
        p=C.props(j['unit']);C.attest_service(j,p)
        if p.get('ActiveState')in C.ACTIVE:return dict(phase='publishing_independent_component',component=c,completed=done)
    for c in ('path','morgan'):
        if done[c]:continue
        source=(P.ROOT44 if c=='path'else P.ROOT43)/'COMPLETE.json'
        if not source.exists():states[c]=dict(phase='waiting_own_scientific_completion');continue
        j=jobs[c];path=CONTROL/'dispatch'/(c+'.json')
        if path.exists():need(P.read(path)==j,'Publication dispatch changed')
        else:P.write(path,j)
        states[c]=C.ensure(j,D)
        if states[c]['phase']in ('running','resumed_or_launched'):break
    return dict(phase='waiting_or_publishing_independent_components',completed=done,decisions=states,gpu_jobs=0)


def preflight():
    total=0
    for pattern in ('test_publisher.py','test_controller.py'):
        r=subprocess.run([PYTHON,'-m','unittest','discover','-s',str(HERE),'-p',pattern,'-v'],capture_output=True,text=True)
        need(r.returncode==0,'Publisher46 tests failed: '+r.stdout+r.stderr);match=re.search(r'Ran (\d+) tests',r.stderr);need(match,'Executed test count missing');total+=int(match.group(1))
    result=dict(schema='publication46-tested-code-v1',phase='complete',tests_passed=total,sources_sha256=sources(),production_git_actions=0,
      components_independent=True,controller_cpu=.1,controller_memory_bytes=256*1024**2,publisher_cpu=2,publisher_memory_bytes=GIB,gpu_jobs=0)
    path=ROOT/'CODE_READY.json'
    if path.exists():need(P.read(path)==result,'Existing publisher readiness changed')
    else:P.write(path,result)
    return result


def main():
    a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');a.add_argument('--once',action='store_true');args=a.parse_args();CONTROL.mkdir(parents=True,exist_ok=True)
    if args.preflight:print(preflight());return
    with (CONTROL/'controller.lock').open('a+')as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (CONTROL/'FAILED.json').exists():return
        try:
            while True:
                value=dict(cycle(),updated_utc=P.now());P.write(CONTROL/'status.json',value)
                if args.once or value['phase']=='complete':return
                time.sleep(30)
        except Exception as error:
            P.write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),scientific_jobs_stopped=0,device_leases_changed=0,updated_utc=P.now()));raise

if __name__=='__main__':main()
