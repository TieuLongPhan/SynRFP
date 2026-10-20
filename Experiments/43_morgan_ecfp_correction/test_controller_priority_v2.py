"""Governor-only priority proof mutations; no service or scientific actions."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('priority43_test_target',HERE/'controller_priority_v2.py')
T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
P=T.P

class PriorityTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.priority=self.root/'priority_v2';self.r41=self.root/'41'
        for obj,key,value in ((P,'ROOT',self.root),(P,'PRIORITY',self.priority),(P,'ROOT41',self.r41),(P,'_CACHE',{})):
            p=patch.object(obj,key,value);p.start();self.addCleanup(p.stop)
        self.source=self.root/'source.py';self.source.write_text('retainedsource')
        self.own={str(self.source):P.sha(self.source)}
        p=patch.object(P,'sources',return_value=self.own);p.start();self.addCleanup(p.stop)
        original=self.put(self.r41/'control_priority_v2/execution.json',dict(sources_sha256=self.own))
        original43=self.put(self.root/'control/execution.json',dict(original=True))
        self.unit_file=self.root/'41.service';self.unit_file.write_text('ExecStart=/usr/bin/taskset -c 31 /python -u /41.py\n')
        self.workers={'synrfp-morgan43-c0-g0001-a.service':dict(MainPID='12',ControlGroup='/morgan',ExecStart='unchangedmorgan',FreezerState='running'),
                      'synrfp-syncluster41-native-ec.service':dict(MainPID='13',ControlGroup='/ec',ExecStart='unchangedec',FreezerState='running')}
        self.intent=dict(phase='approved',unit=P.UNIT,unit_file=str(self.unit_file),unit_sha256=P.sha(self.unit_file),
            prior_enabled=True,prior_active=True,sources_sha256={**self.own,str(original):P.sha(original)},workers_before=self.workers)
        ip=self.put(self.priority/'INTENT.json',self.intent)
        self.proof=dict(phase='complete',intent_sha256=P.sha(ip),governor_only=True,workers_preserved=True,
                       scientific_stop_commands=0,GPU_lease_changes=0,workers_after=copy.deepcopy(self.workers),
                       outputs_sha256={str(ip):P.sha(ip),str(self.unit_file):P.sha(self.unit_file)})
        cp=self.put(self.priority/'PAUSE41_COMPLETE.json',self.proof)
        rp=self.put(self.priority/'CODE_READY.json',dict(phase='complete',tests_passed=1,sources_sha256=self.own))
        review=dict(schema='morgan43-priority-v2-root-reviewed',phase='reviewed',code_ready_sha256=P.sha(rp),
            intent_sha256=P.sha(ip),pause41_complete_sha256=P.sha(cp),original43_execution_sha256=P.sha(original43),
            sources_sha256={**self.own,**self.intent['sources_sha256']})
        self.put(self.priority/'ROOT_REVIEW.json',review)
        self.governor=dict(LoadState='loaded',ActiveState='inactive',MainPID='0',
            ExecStart='{ argv[]=/usr/bin/taskset -c 31 /python -u /41.py ; ignore_errors=no ; }',
            CPUQuotaPerSecUSec='100ms',MemoryMax=str(256*1024**2),WorkingDirectory=str(P.REPO),
            KillMode='control-group',Environment='CUDA_VISIBLE_DEVICES=',Result='success')
        self.live={name:{**props,'ActiveState':'active','Result':'success'} for name,props in self.workers.items()}
        self.old43=dict(ActiveState='inactive',MainPID='0')
        self.value=dict(units=[dict(unit=name,cpu_equivalents=2,memory_max_bytes=8*1024**3,main_pid=int(props['MainPID'])) for name,props in self.workers.items()])
        p=patch.object(P,'props',side_effect=lambda u:self.governor if u==P.UNIT else self.old43 if u=='synrfp-morgan43-v1.service' else self.live[u]);p.start();self.addCleanup(p.stop)
    def put(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    def test_complete_governor_pause_keeps_all_live_workers_counted(self):
        self.assertTrue(P.paused41(self.value))
    def test_absent_review_never_releases_old_reserve(self):
        (self.priority/'ROOT_REVIEW.json').unlink();self.assertFalse(P.paused41(self.value))
    def test_still_running41_governor_never_releases_reserve(self):
        self.governor.update(ActiveState='active',MainPID='20');self.assertFalse(P.paused41(self.value))
    def test_old43_governor_restart_never_releases_reserve(self):
        self.old43.update(ActiveState='active',MainPID='99');self.assertFalse(P.paused41(self.value))
    def test_worker_pid_or_cgroup_change_refused(self):
        unit=next(iter(self.live))
        for field in ('MainPID','ControlGroup','ExecStart'):
            old=self.live[unit][field];self.live[unit][field]='changed'
            with self.assertRaises(ValueError):P.paused41(self.value)
            self.live[unit][field]=old
    def test_transient_systemd_metadata_does_not_change_worker_command(self):
        before='{ path=/python ; argv[]=/python -u /worker.py ; ignore_errors=no ; start_time=[n/a] ; pid=0 ; }'
        after='{ path=/python ; argv[]=/python -u /worker.py ; ignore_errors=no ; start_time=[Tue] ; pid=12 ; }'
        self.assertEqual(P.command(before),P.command(after))
        self.assertNotEqual(P.command(before),P.command(after.replace('/worker.py','/different.py')))
    def test_frozen_worker_refused(self):
        self.live[next(iter(self.live))]['FreezerState']='frozen'
        with self.assertRaises(ValueError):P.paused41(self.value)
    def test_normally_finished_worker_allowed_without_claiming_task_complete(self):
        unit=next(iter(self.live));self.live[unit].update(ActiveState='inactive',MainPID='0',Result='success')
        self.value['units']=[x for x in self.value['units'] if x['unit']!=unit]
        self.assertTrue(P.paused41(self.value))
    def test_failed_worker_does_not_authorize_priority_release(self):
        self.live[next(iter(self.live))].update(ActiveState='failed',MainPID='0',Result='exit-code')
        with self.assertRaises(ValueError):P.paused41(self.value)
    def test_missing_or_unlimited_worker_reservation_refused(self):
        self.value['units'].pop()
        with self.assertRaises(ValueError):P.paused41(self.value)
    def test_unrecorded41_dispatch_refused(self):
        self.value['units'].append(dict(unit='synrfp-syncluster41-unexpected.service',cpu_equivalents=2,memory_max_bytes=8*1024**3))
        with self.assertRaises(ValueError):P.paused41(self.value)
    def test_modified_old_source_refused(self):
        self.source.write_text('changedsource')
        with self.assertRaises(ValueError):P.paused41(self.value)
    def test_changed_unit_file_refused(self):
        self.unit_file.write_text('ExecStart=/different\n')
        with self.assertRaises(ValueError):P.paused41(self.value)
    def test_changed_governor_quota_refused(self):
        self.governor['MemoryMax']='999'
        with self.assertRaises(ValueError):P.paused41(self.value)
    def test_wrong_review_or_proof_binding_refused(self):
        p=self.priority/'ROOT_REVIEW.json';r=P.read(p);r['pause41_complete_sha256']='different';self.put(p,r)
        with self.assertRaises(ValueError):P.paused41(self.value)
    def test_adapter_falls_back_to_original_context_without_proof(self):
        with patch.object(P,'paused41',return_value=False),patch.object(T,'ORIGINAL_CONTEXT',return_value={'active':True}) as original:
            self.assertEqual(T.context(self.value),{'active':True});original.assert_called_once()
    def test_adapter_relaxes_only_future41_reserve_after_proof(self):
        with patch.object(P,'paused41',return_value=True):
            v=T.context(self.value);self.assertFalse(v['active']);self.assertEqual(v['pending_memory_bytes'],0)
    def test_adapter_uses_original_worker_dispatch_and_lock(self):
        text=(HERE/'controller_priority_v2.py').read_text()
        self.assertIn("ROOT/'control/controller.lock'",text)
        self.assertIn('result=O.cycle()',text)
        self.assertNotIn('O.request_yield(',text)
        self.assertNotIn('lease_available(',text)

if __name__=='__main__':unittest.main()
