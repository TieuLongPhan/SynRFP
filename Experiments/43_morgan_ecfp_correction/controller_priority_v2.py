"""Preserve43 execution while honoring the reviewed Morgan/Path CPU priority."""
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
O=module('priority43_original_controller',HERE/'controller.py')
P=module('priority43_governor_pause_policy',HERE/'priority_policy_v2.py')
ROOT=O.ROOT;CONTROL=ROOT/'control_priority_v2';PRIORITY=ROOT/'priority_v2'
ORIGINAL_CONTEXT=O.context
ORIGINAL_UNIT='synrfp-morgan43-v1.service'

def context(value):
    if P.paused41(value):return dict(active=False,pending_cpu=0,pending_memory_bytes=0,reason='reviewed41_future_dispatch_paused')
    return ORIGINAL_CONTEXT(value)
O.context=context

def manifest(value):
    admitted=P.verify_pause(value)
    if admitted is None:return None
    execution=O.manifest()
    if execution is None:return None
    props=O.C.props(ORIGINAL_UNIT)
    O.need(props.get('ActiveState') not in O.C.ACTIVE and props.get('MainPID','0')=='0','Original43 governor is still live')
    expected=dict(schema='morgan43-priority-controller-execution-v2',phase='ready',
        original_execution_sha256=O.B.sha(ROOT/'control/execution.json'),
        sources_sha256={**P.sources(),**execution['sources_sha256'],**admitted['outputs_sha256']},
        original_controller_lock=str(ROOT/'control/controller.lock'),scientific_sources_unchanged=True,
        old_dispatches_retained=True,worker_and_gpu_lease_mutations=0,policy_fraction=.85)
    O.immutable(CONTROL/'execution.json',expected)
    return expected

def preflight():
    p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_controller_priority_v2.py','-v'],capture_output=True,text=True)
    O.need(p.returncode==0,'Priority43 tests failed: '+p.stdout+p.stderr)
    match=re.search(r'Ran (\d+) tests',p.stderr);O.need(match is not None,'No executed priority43 tests')
    value=dict(schema='morgan43-priority-code-ready-v2',phase='complete',tests_passed=int(match.group(1)),sources_sha256=P.sources())
    return O.immutable(PRIORITY/'CODE_READY.json',value)

def cycle():
    value=O.snapshot();execution=manifest(value)
    if execution is None:return dict(phase='waiting_reviewed_governor_only41_pause')
    O.pins(execution['sources_sha256'])
    result=O.cycle()
    O.write(O.CONTROL/'status.json',{**result,'updated_utc':O.B.now(),'operational_priority_adapter':'v2'})
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    if a.preflight:print(json.dumps(preflight()));return
    # This is deliberately the original controller lock and original execution.
    with (ROOT/'control/controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (CONTROL/'FAILED.json').exists() or (ROOT/'control/FAILED.json').exists():return
        try:
            while True:
                value={**cycle(),'updated_utc':O.B.now()};O.write(CONTROL/'status.json',value)
                if a.once or value.get('phase')=='complete':return
                time.sleep(30)
        except Exception as error:
            # An unresolved new operational proof cannot authorize signalling
            # any already-running scientific worker. Stop new dispatch only.
            O.write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),yield_requested=[],
                scientific_artifacts_preserved=True,other_task_mutations=0,updated_utc=O.B.now()))
            raise

if __name__=='__main__':main()
