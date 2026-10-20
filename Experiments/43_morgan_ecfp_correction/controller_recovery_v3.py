"""Recover only the failed Morgan guardian; retain all old science and latches."""
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
V2=module('recovery43_original_priority_v2',HERE/'controller_priority_v2.py')
R=module('recovery43_finite_restart_snapshot',HERE/'resource_snapshot_v3.py')
O=V2.O
ROOT=O.ROOT;CONTROL=ROOT/'control_recovery_v3';REVIEW_ROOT=ROOT/'recovery_v3'
OWN=('controller_recovery_v3.py','resource_snapshot_v3.py','restore41_recovery_v3.py','test_recovery_v3.py',
     'RECOVERY_V3.md','synrfp-morgan43-recovery-v3.service','synrfp-native41-restore-after43-v3.service')

# This is an in-memory operational input adapter, with its own tested source
# seal. No byte in the frozen original controllers/budget is changed.
O.RAW_SNAPSHOT=lambda:R.snapshot(O.B)

def sources():return {str((HERE/n).resolve()):O.B.sha(HERE/n)for n in OWN}

def sealed_execution():
    """Replay the reviewed seal after41 was enabled, without re-proving pause."""
    path=CONTROL/'execution.json'
    if not path.exists():return None
    value=O.read(path);O.pins(value.get('sources_sha256'))
    required={**sources(),str(REVIEW_ROOT/'CODE_READY.json'):O.B.sha(REVIEW_ROOT/'CODE_READY.json'),
        str(REVIEW_ROOT/'ROOT_REVIEW.json'):O.B.sha(REVIEW_ROOT/'ROOT_REVIEW.json'),
        str(ROOT/'control_priority_v2/FAILED.json'):O.B.sha(ROOT/'control_priority_v2/FAILED.json')}
    O.need(value.get('schema')=='morgan43-recovery-execution-v3' and value.get('phase')=='ready'
        and value.get('original_execution_sha256')==O.B.sha(ROOT/'control/execution.json')
        and value.get('preserved_failure_sha256')==O.B.sha(ROOT/'control_priority_v2/FAILED.json')
        and value.get('original_lock')==str(ROOT/'control/controller.lock')
        and value.get('scientific_sources_unchanged') is True and value.get('worker_signal_changes')==0
        and value.get('gpu_lease_changes')==0
        and all(value['sources_sha256'].get(p)==h for p,h in required.items()),'Changed recovery execution seal')
    return value

def manifest():
    paths={n:REVIEW_ROOT/n for n in ('CODE_READY.json','ROOT_REVIEW.json')}
    if not all(p.exists()for p in paths.values()):return None
    ready,review=(O.read(paths[n])for n in paths)
    O.need(ready.get('phase')=='complete' and ready.get('tests_passed',0)>0 and ready.get('sources_sha256')==sources(),
           'Morgan recovery code readiness changed')
    O.pins(ready['sources_sha256']);O.pins(review.get('sources_sha256'))
    failure=ROOT/'control_priority_v2/FAILED.json'
    O.need(review.get('schema')=='morgan43-recovery-v3-root-reviewed' and review.get('phase')=='reviewed'
        and review.get('code_ready_sha256')==O.B.sha(paths['CODE_READY.json'])
        and review.get('preserved_failure_sha256')==O.B.sha(failure)
        and review.get('original_execution_sha256')==O.B.sha(ROOT/'control/execution.json')
        and review.get('restore35_via_original_verified_mlp_gate') is True
        and review.get('restore41_after_both_reports_and_guardian_exit') is True,'Recovery review lacks exact failed/current execution binding')
    reason=O.read(failure).get('reason','')
    O.need(reason=="IntegrityError('Resource audit invalid for active research unit: synrfp-task-preset44-parallel-v3.service')",
           'Morgan recovery review names another unresolved failure')
    for unit in ('synrfp-morgan43-v1.service','synrfp-morgan43-priority-v2.service'):
        p=O.C.props(unit);O.need(p.get('ActiveState') not in O.C.ACTIVE and p.get('MainPID','0')=='0','Previous Morgan guardian is still live')
    value=O.snapshot();old=V2.manifest(value)
    if old is None:return None
    required={**old['sources_sha256'],**sources()}
    O.need(all(review['sources_sha256'].get(p)==h for p,h in required.items()),'Recovery review omits original or new source')
    execution=dict(schema='morgan43-recovery-execution-v3',phase='ready',
        original_execution_sha256=O.B.sha(ROOT/'control/execution.json'),preserved_failure_sha256=O.B.sha(failure),
        sources_sha256={**required,str(failure):O.B.sha(failure),**{str(p):O.B.sha(p)for p in paths.values()}},
        original_lock=str(ROOT/'control/controller.lock'),scientific_sources_unchanged=True,worker_signal_changes=0,gpu_lease_changes=0)
    O.immutable(CONTROL/'execution.json',execution);return execution

def cycle():
    execution=manifest()
    if execution is None:return dict(phase='waiting_exact_recovery_review')
    O.pins(execution['sources_sha256'])
    value=O.cycle()
    O.write(O.CONTROL/'status.json',{**value,'operational_adapter':'recovery-v3','updated_utc':O.B.now()})
    return value

def preflight():
    p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_recovery_v3.py','-v'],capture_output=True,text=True)
    O.need(p.returncode==0,'Recovery controls failed: '+p.stdout+p.stderr)
    m=re.search(r'Ran (\d+) tests',p.stderr);O.need(m,'No recovery test count')
    value=dict(schema='morgan43-recovery-code-ready-v3',phase='complete',tests_passed=int(m.group(1)),sources_sha256=sources())
    return O.immutable(REVIEW_ROOT/'CODE_READY.json',value)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    if a.preflight:print(preflight());return
    with(ROOT/'control/controller.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(CONTROL/'FAILED.json').exists()or(ROOT/'control/FAILED.json').exists():return
        try:
            while True:
                value={**cycle(),'updated_utc':O.B.now()};O.write(CONTROL/'status.json',value)
                if a.once or value.get('phase')=='complete':return
                time.sleep(30)
        except Exception as error:
            O.write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),old_failure_preserved=True,
                scientific_signals_sent=0,scientific_artifacts_preserved=True,updated_utc=O.B.now()));raise

if __name__=='__main__':main()
