"""New42-only85% admission; sealed older controllers retain their75% policy."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
s=importlib.util.spec_from_file_location('knn42_original_budget',REPO/'Experiments/25_followup_scheduler/budget.py')
LEGACY=importlib.util.module_from_spec(s);s.loader.exec_module(LEGACY)
GIB=1024**3
FRACTION=.85
PRIORITY_ROOT=Path.home()/'.local/share/synrfp-runs/syncluster_main12_v1'


def effective(snapshot):
    return {**snapshot,'cpu_budget_equivalents':snapshot['physical_cores']*FRACTION,
            'memory_budget_bytes':int(snapshot['memory_total_bytes']*FRACTION),
            'policy_fraction':FRACTION,'global_available_memory_floor_fraction':.15,
            'legacy75_controllers_unchanged':True}

def snapshot(raw_snapshot=None,artifact_root=None):
    # Capture the original function through an isolated module. Never assign a
    # wrapper to LEGACY.snapshot, avoiding recursive cross-controller patches.
    raw=LEGACY.snapshot() if raw_snapshot is None else raw_snapshot()
    result=effective(raw)
    if artifact_root is not None:
        p=Path(artifact_root)
        while not p.exists():p=p.parent
        result['artifact_disk_free_bytes']=shutil.disk_usage(p.resolve()).free
        result['disk_free_bytes']=min(result['disk_free_bytes'],result['artifact_disk_free_bytes'])
    return result

def gpu_headroom(value):
    # Fixed ADDITIONAL reserve: another35dispatch may race this snapshot.
    # Current35 quotas remain fully counted in aggregate caps.
    return 2.,4*GIB

def admission(value,cpu,memory_bytes,priority=None):
    reasons=[];cpu_limit=value['physical_cores']*.85;memory_limit=int(value['memory_total_bytes']*.85)
    if value['cpu_cap_sum']+cpu>cpu_limit+1e-8:reasons.append('aggregate_cpu_85pct')
    if value['memory_cap_sum_bytes']+memory_bytes>memory_limit:reasons.append('aggregate_memory_85pct')
    if value['memory_available_bytes']<value['memory_total_bytes']*.15:reasons.append('global_available_ram_15pct')
    if value['disk_free_bytes']<40*GIB:reasons.append('disk_reserve_40GiB')
    gpu_cpu,gpu_memory=gpu_headroom(value)
    reserve_cpu=gpu_cpu;reserve_memory=gpu_memory
    if priority and priority.get('active'):
        # Older41 still uses75% caps/25% available RAM. Reserve its next-stage
        # demand before new42, even though new42's absolute ceiling is85%.
        reserve_cpu+=priority.get('pending_cpu',2)
        reserve_memory+=priority.get('pending_memory_bytes',24*GIB)
        if value['cpu_cap_sum']+cpu+reserve_cpu>value['physical_cores']*.75+1e-8:reasons.append('clustering_priority_cpu_headroom')
        if value['memory_cap_sum_bytes']+memory_bytes+reserve_memory>int(value['memory_total_bytes']*.75):reasons.append('clustering_priority_memory_headroom')
        if value['memory_available_bytes']<value['memory_total_bytes']*.35:reasons.append('clustering_priority_ram_launch_35pct')
    else:
        if value['cpu_cap_sum']+cpu+reserve_cpu>cpu_limit+1e-8:reasons.append('gpu35_cpu_headroom')
        if value['memory_cap_sum_bytes']+memory_bytes+reserve_memory>memory_limit:reasons.append('gpu35_memory_headroom')
    return reasons

def pressure(value,priority_active,previous=False):
    low=.30 if priority_active else .15;recover=.35 if priority_active else .20
    ratio=value['memory_available_bytes']/value['memory_total_bytes']
    low_disk=value['disk_free_bytes']<44*GIB
    held=ratio<low or low_disk or (previous and (ratio<recover or value['disk_free_bytes']<48*GIB))
    return dict(yield_required=held,critical_freeze=ratio<(.25 if priority_active else .15) or value['disk_free_bytes']<40*GIB,
        thaw_allowed=ratio>=recover and value['disk_free_bytes']>=48*GIB,
        available_memory_fraction=ratio,low_threshold=low,recovery_threshold=recover)
