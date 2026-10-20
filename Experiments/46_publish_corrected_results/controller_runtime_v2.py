"""Reviewed governor-only handoff to isolated46v2 five-lane-aware policy."""
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
    q=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(q);sys.modules[name]=m;q.loader.exec_module(m);return m

V=module('publication46_v2_policy',HERE/'publisher_runtime_v2.py')
O=module('publication46_v2_frozen_controller',HERE/'controller.py')
ROOT=V.ROOT
CONTROL=ROOT/'control'
OLD_UNIT='synrfp-corrected-results-publish46-v1.service'
O.ROOT=ROOT;O.CONTROL=CONTROL;O.P=V.P;O.Q.ROOT=CONTROL;O.C.ROOT=CONTROL/'managed';O.B.ROOT=O.C.ROOT
ORIGINAL_SPEC=O.spec
ORIGINAL_REVIEW=O.review


def spec(component,review):
    j=ORIGINAL_SPEC(component,review)
    j.update(id='publication46v2-'+component,unit='synrfp-publish46v2-'+component+'-'+V.P.sha(ROOT/'config.json')[:8],
      arguments=[str(HERE/'publisher_runtime_v2.py'),'--component',component])
    return j


def review():
    r=ORIGINAL_REVIEW()
    if r is None:return None
    config=V.P.read(ROOT/'config.json')
    O.need(config.get('schema')==V.SCHEMA and config.get('publication_runtime_version')==2
      and config.get('required_path_v3_sources_sha256')==V.require_v3_sources(),'Wrong five-lane publication policy')
    O.need(r.get('publication_runtime_version')==2 and r.get('supersedes_publication_root')==str(V.LEGACY_ROOT)
      and r.get('governor_only_handoff')is True,'Version2 review must authorize the governor-only handoff')
    return r


def old_publication_idle():
    props=O.C.props(OLD_UNIT)
    if props.get('ActiveState')in O.C.ACTIVE or props.get('MainPID','0')!='0':return False
    for path in (V.LEGACY_ROOT/'control/dispatch').glob('*.json'):
        j=V.P.read(path);O.need(j.get('id')in ('publication46-path','publication46-morgan'),'Foreign prior publication job')
        state=O.C.props(j['unit']);O.C.attest_service(j,state)
        if state.get('ActiveState')in O.C.ACTIVE or state.get('MainPID','0')!='0':return False
    return True


O.spec=spec;O.review=review


def cycle():
    r=review()
    if r is None:return dict(phase='waiting_reviewed_v2_five_lane_publication_policy')
    if not old_publication_idle():return dict(phase='waiting_original46_governor_and_publication_workers_idle')
    return O.cycle()


def preflight():
    total=0
    for pattern in ('test_publisher.py','test_controller.py','test_runtime_v2.py'):
        result=subprocess.run([O.PYTHON,'-m','unittest','discover','-s',str(HERE),'-p',pattern,'-v'],capture_output=True,text=True)
        O.need(result.returncode==0,'Publication46v2 tests failed: '+result.stdout+result.stderr);match=re.search(r'Ran (\d+) tests',result.stderr);O.need(match,'No executed test count');total+=int(match.group(1))
    value=dict(schema='publication46-runtime-v2-tested-code',phase='complete',tests_passed=total,sources_sha256=O.sources(),publication_runtime_version=2,
      production_git_actions=0,scientific_gates_unchanged=True,components_independent=True,gpu_jobs=0)
    path=ROOT/'CODE_READY.json'
    if path.exists():O.need(V.P.read(path)==value,'Existing46v2 tested readiness changed')
    else:V.P.write(path,value)
    return value


def main():
    a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');a.add_argument('--once',action='store_true');args=a.parse_args();CONTROL.mkdir(parents=True,exist_ok=True)
    if args.preflight:print(preflight());return
    with (CONTROL/'controller.lock').open('a+')as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (CONTROL/'FAILED.json').exists():return
        try:
            while True:
                value=dict(cycle(),updated_utc=V.P.now());V.P.write(CONTROL/'status.json',value)
                if args.once or value['phase']=='complete':return
                time.sleep(30)
        except Exception as error:
            V.P.write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),scientific_jobs_stopped=0,device_leases_changed=0,updated_utc=V.P.now()));raise

if __name__=='__main__':main()
