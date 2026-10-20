"""A reviewed governor-only41 pause releases future, not live, reservations.

This operational policy has no scientific imports, service mutations or CUDA
access. Both the versioned43 adapter and the new44 queue use the same proof.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path.home()/'.local/share/synrfp-runs/morgan_ecfp_correction_v1'
ROOT41=ROOT.parent/'syncluster_main12_v1'
PRIORITY=ROOT/'priority_v2'
UNIT='synrfp-syncluster41-priority-v2.service'
ACTIVE={'active','activating','reloading','deactivating'}
_CACHE={}

def need(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(p,h):
    p=Path(p);s=p.stat();k=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    prior=_CACHE.get(str(p.resolve()))
    if prior is None or prior[0]!=k:
        digest=sha(p);after=p.stat()
        need(k==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Priority input changed while hashing')
        _CACHE[str(p.resolve())]=(k,digest)
    need(_CACHE[str(p.resolve())][1]==h,'Priority source/hash mismatch: '+str(p))
def pins(v):
    need(isinstance(v,dict) and bool(v),'Missing priority source catalogue')
    for p,h in v.items():need(Path(p).is_absolute(),'Priority binding is not absolute');pin(p,h)
def props(unit):
    keys='LoadState,ActiveState,MainPID,ControlGroup,FreezerState,ExecStart,Environment,WorkingDirectory,CPUQuotaPerSecUSec,MemoryMax,KillMode,Result'
    v=subprocess.run(['systemctl','--user','show',unit,'--property='+keys],capture_output=True,text=True,check=True)
    return dict(x.split('=',1) for x in v.stdout.splitlines() if '=' in x)
def sources():
    names=('priority_policy_v2.py','controller_priority_v2.py','test_controller_priority_v2.py','synrfp-morgan43-priority-v2.service')
    return {str((HERE/n).resolve()):sha(HERE/n) for n in names}

def command(value):
    """Systemd execution metadata may change while the executable argv does not."""
    match=re.search(r'argv\[\]=(.*?) ; ignore_errors=',value or '')
    if match:return tuple(shlex.split(match.group(1)))
    need(not (value or '').lstrip().startswith('{'),'Malformed systemd command witness')
    return tuple(shlex.split(value or ''))

def attest_governor(intent,properties):
    need(intent['unit']==UNIT,'Wrong41 governor pause scope')
    unit_file=Path(intent['unit_file']);pin(unit_file,intent['unit_sha256'])
    lines=[x.split('=',1)[1] for x in unit_file.read_text().splitlines() if x.startswith('ExecStart=')]
    match=re.search(r'argv\[\]=(.*?) ; ignore_errors=',properties.get('ExecStart',''))
    actual=shlex.split(match.group(1)) if match else []
    need(len(lines)==1 and actual==shlex.split(lines[0]),'Paused41 governor command changed')
    need(properties.get('LoadState')=='loaded' and properties.get('CPUQuotaPerSecUSec') in ('100ms','100000us','0.1s')
         and properties.get('MemoryMax')==str(256*1024**2) and properties.get('WorkingDirectory')==str(REPO)
         and properties.get('KillMode')=='control-group','Paused41 governor caps/context changed')
    env=dict(x.split('=',1) for x in shlex.split(properties.get('Environment','')))
    need(env.get('CUDA_VISIBLE_DEVICES')=='','Paused41 governor CUDA environment changed')

def verify_pause(value):
    """Return the admitted proof, or None before the audited pause exists.

    An absent proof never relaxes policy. A malformed or changed published proof
    fails closed. Existing41 workers stay counted in the actual cgroup snapshot.
    """
    paths={n:PRIORITY/n for n in ('CODE_READY.json','ROOT_REVIEW.json','INTENT.json','PAUSE41_COMPLETE.json')}
    if not all(p.exists() for p in paths.values()):return None
    ready,review,intent,proof=(read(paths[n]) for n in paths)
    need(ready.get('phase')=='complete' and ready.get('tests_passed',0)>0 and ready.get('sources_sha256')==sources(),
         'Untested or changed43 priority adapter')
    pins(ready['sources_sha256']);pins(intent.get('sources_sha256'));pins(review.get('sources_sha256'))
    need(review.get('schema')=='morgan43-priority-v2-root-reviewed' and review.get('phase')=='reviewed'
         and review.get('code_ready_sha256')==sha(paths['CODE_READY.json'])
         and review.get('intent_sha256')==sha(paths['INTENT.json'])
         and review.get('pause41_complete_sha256')==sha(paths['PAUSE41_COMPLETE.json'])
         and review.get('original43_execution_sha256')==sha(ROOT/'control/execution.json'),
         'Priority43 root review does not bind the tested adapter and exact drain')
    need(all(review['sources_sha256'].get(p)==h for p,h in {**ready['sources_sha256'],**intent['sources_sha256']}.items()),
         'Priority43 root review omits tested/original sources')
    need(intent.get('phase')=='approved' and intent.get('unit')==UNIT and intent.get('prior_enabled') is True
         and intent.get('prior_active') is True,
         'Priority41 pause intent exceeds governor-only scope')
    original41=ROOT41/'control_priority_v2/execution.json'
    pin(original41,intent['sources_sha256'][str(original41)])
    original=read(original41);pins(original['sources_sha256'])
    need(all(intent['sources_sha256'].get(p)==h for p,h in original['sources_sha256'].items()),
         'Pause intent omits original41 execution sources')
    need(proof.get('phase')=='complete' and proof.get('intent_sha256')==sha(paths['INTENT.json'])
         and proof.get('governor_only') is True and proof.get('workers_preserved') is True,
         'No completed worker-preserving41 governor drain')
    need(proof.get('scientific_stop_commands')==0 and proof.get('GPU_lease_changes')==0,
         'Priority drain changed science workers or GPU leases')
    pins(proof.get('outputs_sha256'))
    before,after=intent.get('workers_before'),proof.get('workers_after')
    need(isinstance(before,dict) and isinstance(after,dict) and set(before)==set(after),'Wrong41 worker preservation population')
    for unit,old in before.items():
        need(unit.startswith(('synrfp-syncluster41-','synrfp-morgan43-')) and unit!=UNIT,'Wrong science worker in pause intent')
        need(all(old.get(k)==after[unit].get(k) for k in ('MainPID','ControlGroup'))
             and command(old.get('ExecStart'))==command(after[unit].get('ExecStart')),
             '41 worker changed during governor-only drain')
        current=props(unit)
        if current.get('ActiveState') in ACTIVE:
            need(all(current.get(k)==old.get(k) for k in ('MainPID','ControlGroup'))
                 and command(current.get('ExecStart'))==command(old.get('ExecStart'))
                 and current.get('FreezerState') in ('running','thawed',''), '41 live worker PID/cgroup changed or frozen')
            matches=[x for x in value.get('units',[]) if x['unit'].removesuffix('.service')==unit.removesuffix('.service')]
            need(len(matches)==1 and matches[0]['cpu_equivalents']>0 and matches[0]['memory_max_bytes']>0
                 and str(matches[0].get('main_pid'))==current.get('MainPID'),
                 'Preserved science worker is absent from finite-cgroup accounting')
        else:
            need(current.get('ActiveState')=='inactive' and current.get('MainPID','0')=='0' and current.get('Result')=='success',
                 '41 worker did not remain live or finish normally')
    exempt={UNIT.removesuffix('.service'),'synrfp-syncluster41-cpu-priority-v2','synrfp-syncluster41-input-repair-v2'}
    for unit in value.get('units',[]):
        name=unit['unit'].removesuffix('.service')
        if not name.startswith('synrfp-syncluster41-') or name in exempt:continue
        need(unit['unit'] in before or name+'.service' in before,'Unrecorded41 science dispatch appeared during pause')
        need(unit['cpu_equivalents']>0 and unit['memory_max_bytes']>0,'Live41 worker is not counted by finite caps')
    p=props(UNIT);attest_governor(intent,p)
    if p.get('ActiveState')!='inactive' or p.get('MainPID','0')!='0':return None
    old43=props('synrfp-morgan43-v1.service')
    if old43.get('ActiveState') in ACTIVE or old43.get('MainPID','0')!='0':return None
    return dict(intent=intent,proof=proof,review=review,
                outputs_sha256={str(p):sha(p) for p in paths.values()})

def paused41(value):return verify_pause(value) is not None
