"""Restore the paused41 governor only after both reports and guardian exit."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
import time

import controller_recovery_v3 as G

O=G.O
ROOT=O.ROOT
CONTROL=ROOT/'restore41_recovery_v3'
F=G.module('recovery43_original_path_restore',O.REPO/'Experiments/44_task_preset_comparison/controller.py')
F.P43=G.V2.P
F.snapshot=lambda:O.snapshot()
F.B.snapshot=F.snapshot

def cycle():
    # Validate the same immutable operational source review without requiring
    # the completed guardian itself to remain active.
    if (G.sealed_execution()or G.manifest()) is None:return dict(phase='waiting_recovery_review')
    if not F.morgan_report_complete():return dict(phase='waiting_Morgan_full_report')
    if not F.report_complete():return dict(phase='waiting_Path_full_report')
    p=O.C.props('synrfp-morgan43-recovery-v3.service')
    if p.get('ActiveState') in O.C.ACTIVE or p.get('MainPID','0')!='0':return dict(phase='waiting_recovery_guardian_exit')
    # F.restore41 already verifies all old43 services and the native closures.
    # Add every new44 v3/v4 service before passing its empty old-job list.
    value=O.snapshot()
    live=[u for u in value['units']if u['unit'].startswith(('synrfp-preset44-','synrfp-task-preset44-'))]
    if live:return dict(phase='waiting_all_Path_science_report_and_guardian_exit',units=[u['unit']for u in live])
    result=F.restore41([])
    return dict(phase='complete'if result['phase']=='complete'else'waiting_original41_restore_gate',restore=result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args();CONTROL.mkdir(parents=True,exist_ok=True)
    with(CONTROL/'controller.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(CONTROL/'FAILED.json').exists():return
        try:
            while True:
                value={**cycle(),'updated_utc':O.B.now()};O.write(CONTROL/'status.json',value)
                if a.once or value['phase']=='complete':return
                time.sleep(30)
        except Exception as error:
            O.write(CONTROL/'FAILED.json',dict(phase='failed',reason=repr(error),scientific_artifacts_preserved=True,
                    scientific_signals_sent=0,updated_utc=O.B.now()));raise

if __name__=='__main__':main()
