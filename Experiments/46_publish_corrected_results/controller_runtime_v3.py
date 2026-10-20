"""Persistent independent Morgan/Path publisher after the scoped report repair."""
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
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


V=module('publication46_v3_policy',HERE/'publisher_runtime_v3.py')
O=module('publication46_v3_frozen_controller',HERE/'controller.py')
ROOT=V.ROOT;CONTROL=ROOT/'control'
O.ROOT=ROOT;O.CONTROL=CONTROL;O.P=V.P;O.Q.ROOT=CONTROL;O.C.ROOT=CONTROL/'managed';O.B.ROOT=O.C.ROOT
ORIGINAL_SPEC=O.spec;ORIGINAL_REVIEW=O.review
OLD_UNITS=('synrfp-corrected-results-publish46-v1.service','synrfp-corrected-results-publish46-v2.service')


def spec(component,review):
    job=ORIGINAL_SPEC(component,review)
    job.update(id='publication46v3-'+component,unit='synrfp-publish46v3-'+component+'-'+V.P.sha(ROOT/'config.json')[:8],
      arguments=[str(HERE/'publisher_runtime_v3.py'),'--component',component])
    return job


def review():
    value=ORIGINAL_REVIEW()
    if value is None:return None
    config=V.P.read(ROOT/'config.json')
    O.need(config.get('schema')==V.SCHEMA and config.get('publication_runtime_version')==3
      and config.get('required_path_recovery_sources_sha256')==V.recovery_sources(),'Wrong report-recovery publication policy')
    O.need(value.get('publication_runtime_version')==3 and value.get('supersedes_publication_root')==str(V.LEGACY_ROOT)
      and value.get('governor_only_handoff')is True,'Version3 review must authorize the governor-only handoff')
    return value


def old_publication_idle():
    for unit in OLD_UNITS:
        props=O.C.props(unit)
        if props.get('ActiveState')in O.C.ACTIVE or props.get('MainPID','0')!='0':return False
    for root,ids in ((V.MUTEX_ROOT,('publication46-path','publication46-morgan')),
                     (V.LEGACY_ROOT,('publication46v2-path','publication46v2-morgan'))):
        for path in (root/'control/dispatch').glob('*.json'):
            job=V.P.read(path);O.need(job.get('id')in ids,'Foreign prior publication job')
            state=O.C.props(job['unit']);O.C.attest_service(job,state)
            if state.get('ActiveState')in O.C.ACTIVE or state.get('MainPID','0')!='0':return False
    return True


O.spec=spec;O.review=review


def cycle():
    if review()is None:return dict(phase='waiting_reviewed_v3_report_recovery_publication_policy')
    if not old_publication_idle():return dict(phase='waiting_previous_publication_governors_and_workers_idle')
    return O.cycle()


def preflight():
    total=0
    for pattern in ('test_publisher.py','test_controller.py','test_runtime_v2.py','test_runtime_v3.py'):
        result=subprocess.run([O.PYTHON,'-m','unittest','discover','-s',str(HERE),'-p',pattern,'-v'],capture_output=True,text=True)
        O.need(result.returncode==0,'Publication46v3 tests failed: '+result.stdout+result.stderr)
        match=re.search(r'Ran (\d+) tests',result.stderr);O.need(match,'No executed test count');total+=int(match.group(1))
    value=dict(schema='publication46-runtime-v3-tested-code',phase='complete',tests_passed=total,sources_sha256=O.sources(),publication_runtime_version=3,
      production_git_actions=0,scientific_cells_and_scoring_unchanged=True,components_independent=True,gpu_jobs=0)
    path=ROOT/'CODE_READY.json'
    if path.exists():O.need(V.P.read(path)==value,'Existing46v3 tested readiness changed')
    else:V.P.write(path,value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--once',action='store_true');args=parser.parse_args();CONTROL.mkdir(parents=True,exist_ok=True)
    if args.preflight:print(preflight());return
    with (CONTROL/'controller.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (CONTROL/'FAILED.json').exists():return
        try:
            while True:
                value=dict(cycle(),updated_utc=V.P.now());V.P.write(CONTROL/'status.json',value)
                if args.once or value['phase']=='complete':return
                time.sleep(30)
        except Exception as error:
            V.P.write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),scientific_jobs_stopped=0,device_leases_changed=0,updated_utc=V.P.now()));raise


if __name__=='__main__':main()
