"""Finite-cap accounting across systemd's PID-zero automatic-restart window.

An auto-restart service has no process/cgroup during its restart delay. Count
its configured finite caps conservatively. All other live missing-cgroup cases
remain integrity failures; no actual research reservation is omitted.
"""
from __future__ import annotations
from pathlib import Path
import re
import shutil
import subprocess
import time

ACTIVE={'active','activating','reloading','deactivating'}
CGROUP=Path('/sys/fs/cgroup')
FIELDS='LoadState,ActiveState,SubState,MainPID,ControlGroup,CPUQuotaPerSecUSec,MemoryMax,MemoryCurrent,FragmentPath,InvocationID,Result'

def need(value,message):
    if not value:raise RuntimeError(message)

def properties(unit):
    v=subprocess.run(['systemctl','--user','show',unit,'--property='+FIELDS],capture_output=True,text=True,check=True)
    return dict(x.split('=',1)for x in v.stdout.splitlines()if'='in x)

def cpu_quota(value):
    match=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(us|ms|s)',value or'')
    need(match is not None,'Research restart lacks finite CPU quota')
    quota=float(match.group(1))*{'us':1e-6,'ms':1e-3,'s':1}[match.group(2)]
    need(quota>0,'Research CPU quota must be positive');return quota

def configured_restart(unit,p):
    need(p.get('LoadState')=='loaded' and p.get('ActiveState')=='activating' and p.get('SubState')=='auto-restart'
         and p.get('MainPID')=='0' and not p.get('ControlGroup'),'Unproven research cgroup transition: '+unit)
    cpu=cpu_quota(p.get('CPUQuotaPerSecUSec'))
    memory=p.get('MemoryMax','');need(memory.isdecimal() and int(memory)>0,'Research restart lacks finite memory cap: '+unit)
    return dict(unit=unit,state='activating',cpu_equivalents=cpu,memory_max_bytes=int(memory),
                memory_current_bytes=0,oom_kill_events=0,main_pid=0,cgroup_present=False,
                automatic_restart_pending=True,reservation_origin='configured-finite-caps-during-proven-auto-restart')

def read_unit(unit,retries=3):
    """Bounded retry of a changing unit; stable live failures never get skipped."""
    for attempt in range(retries):
        p=properties(unit)
        if p.get('ActiveState') not in ACTIVE:return None
        group=p.get('ControlGroup','')
        if not group:
            if p.get('ActiveState')=='activating' and p.get('SubState')=='auto-restart' and p.get('MainPID')=='0':
                return configured_restart(unit,p)
            if attempt+1<retries:time.sleep(.05);continue
            raise RuntimeError('Active research unit has no audited cgroup: '+unit)
        need(group.startswith('/') and group!='/' and Path(group).name==unit,'Research cgroup path is not its named service: '+unit)
        cg=CGROUP/group.lstrip('/')
        try:
            quota,period=(cg/'cpu.max').read_text().split();maximum=(cg/'memory.max').read_text().strip()
            need(quota.isdecimal() and period.isdecimal() and int(quota)>0 and int(period)>0,
                 'Active research service lacks finite CPU quota: '+unit)
            need(maximum.isdecimal() and int(maximum)>0,'Active research service lacks finite memory cap: '+unit)
            events=dict(x.split()for x in(cg/'memory.events').read_text().splitlines())
            current=int((cg/'memory.current').read_text())
        except FileNotFoundError:
            after=properties(unit)
            if after.get('ActiveState') not in ACTIVE:return None
            if attempt+1<retries:time.sleep(.05);continue
            raise RuntimeError('Still-active research cgroup missing after bounded retries: '+unit)
        return dict(unit=unit,state=p['ActiveState'],cpu_equivalents=int(quota)/int(period),memory_max_bytes=int(maximum),
                    memory_current_bytes=current,oom_kill_events=int(events.get('oom_kill',0)),main_pid=int(p.get('MainPID','0')),
                    cgroup_present=True,automatic_restart_pending=False)
    raise AssertionError('Unreachable resource retry state')

def snapshot(budget):
    v=subprocess.run(['systemctl','--user','list-units','--type=service','--state=active,activating,reloading,deactivating',
                      '--plain','--no-pager','--no-legend','synrfp*'],capture_output=True,text=True,check=True)
    units=[x.split()[0]for x in v.stdout.splitlines()if x.strip()];records=[]
    need(len(units)==len(set(units)),'Duplicate research service inventory')
    for unit in units:
        need(re.fullmatch(r'synrfp[^/\s]*\.service',unit) is not None,'Foreign research inventory name')
        record=read_unit(unit)
        if record is not None:records.append(record)
    memory=budget.memory();cores=budget.physical_cores()
    return dict(updated_utc=budget.now(),physical_cores=cores,cpu_budget_equivalents=cores*.75,
        memory_total_bytes=memory['MemTotal'],memory_budget_bytes=int(memory['MemTotal']*.75),
        memory_available_bytes=memory['MemAvailable'],cpu_cap_sum=sum(r['cpu_equivalents']for r in records),
        memory_cap_sum_bytes=sum(r['memory_max_bytes']for r in records),disk_free_bytes=shutil.disk_usage(Path.home()).free,
        units=records,resource_audit_profile='finite-cgroups-plus-conservative-proven-auto-restart-v3')
