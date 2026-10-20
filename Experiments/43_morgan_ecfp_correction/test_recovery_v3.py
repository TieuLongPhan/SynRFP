"""Conservative systemd race recovery and preserved scientific/restore gates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=load('morgan43_recovery_test_target',HERE/'controller_recovery_v3.py');R=T.R
U=load('morgan43_restore_test_target',HERE/'restore41_recovery_v3.py')

class ResourceControls(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        p=patch.object(R,'CGROUP',self.root);p.start();self.addCleanup(p.stop)
        p=patch.object(R.time,'sleep');p.start();self.addCleanup(p.stop)
        self.unit='synrfp-example.service';self.group=self.root/'group'/self.unit;self.group.mkdir(parents=True)
        for name,value in [('cpu.max','100000 100000'),('memory.max',str(7*1024**3)),('memory.current','123'),('memory.events','oom_kill 0\n')]:
            (self.group/name).write_text(value)
        self.active=dict(LoadState='loaded',ActiveState='active',SubState='running',MainPID='42',ControlGroup='/group/'+self.unit)
        self.pending=dict(LoadState='loaded',ActiveState='activating',SubState='auto-restart',MainPID='0',ControlGroup='',
                          CPUQuotaPerSecUSec='100ms',MemoryMax=str(256*1024**2))
    def test_real_finite_cgroup_retains_actual_caps(self):
        with patch.object(R,'properties',return_value=self.active):r=R.read_unit(self.unit)
        self.assertEqual((r['cpu_equivalents'],r['memory_max_bytes'],r['main_pid']),(1,7*1024**3,42))
        self.assertTrue(r['cgroup_present'])
    def test_auto_restart_pid_zero_counts_configured_caps_conservatively(self):
        with patch.object(R,'properties',return_value=self.pending):r=R.read_unit(self.unit)
        self.assertEqual((r['cpu_equivalents'],r['memory_max_bytes']),(.1,256*1024**2))
        self.assertTrue(r['automatic_restart_pending']);self.assertFalse(r['cgroup_present'])
    def test_auto_restart_infinite_memory_never_admits(self):
        self.pending['MemoryMax']='infinity'
        with patch.object(R,'properties',return_value=self.pending),self.assertRaises(RuntimeError):R.read_unit(self.unit)
    def test_auto_restart_infinite_cpu_never_admits(self):
        self.pending['CPUQuotaPerSecUSec']='infinity'
        with patch.object(R,'properties',return_value=self.pending),self.assertRaises(RuntimeError):R.read_unit(self.unit)
    def test_running_pid_without_cgroup_remains_integrity_failure(self):
        self.active['ControlGroup']=''
        with patch.object(R,'properties',return_value=self.active)as properties,self.assertRaises(RuntimeError):R.read_unit(self.unit)
        self.assertEqual(properties.call_count,3)
    def test_auto_restart_with_live_pid_is_not_an_exception(self):
        self.pending['MainPID']='12'
        with patch.object(R,'properties',return_value=self.pending),self.assertRaises(RuntimeError):R.read_unit(self.unit)
    def test_other_startup_state_with_no_cgroup_is_not_ignored(self):
        self.pending['SubState']='start-pre'
        with patch.object(R,'properties',return_value=self.pending),self.assertRaises(RuntimeError):R.read_unit(self.unit)
    def test_root_cgroup_cannot_substitute_for_service_cgroup(self):
        self.active['ControlGroup']='/'
        with patch.object(R,'properties',return_value=self.active),self.assertRaises(RuntimeError):R.read_unit(self.unit)
    def test_a_different_named_cgroup_cannot_supply_caps(self):
        self.active['ControlGroup']='/group/synrfp-other.service'
        with patch.object(R,'properties',return_value=self.active),self.assertRaises(RuntimeError):R.read_unit(self.unit)
    def test_active_cgroup_without_finite_memory_is_refused(self):
        (self.group/'memory.max').write_text('max')
        with patch.object(R,'properties',return_value=self.active),self.assertRaises(RuntimeError):R.read_unit(self.unit)
    def test_active_cgroup_without_finite_cpu_is_refused(self):
        (self.group/'cpu.max').write_text('max 100000')
        with patch.object(R,'properties',return_value=self.active),self.assertRaises(RuntimeError):R.read_unit(self.unit)
    def test_only_a_confirmed_naturally_exited_unit_can_disappear(self):
        (self.group/'cpu.max').unlink()
        with patch.object(R,'properties',side_effect=[self.active,dict(ActiveState='inactive',MainPID='0')]):
            self.assertIsNone(R.read_unit(self.unit))
    def test_still_live_disappearing_cgroup_has_bounded_failure(self):
        (self.group/'cpu.max').unlink()
        with patch.object(R,'properties',return_value=self.active)as props,self.assertRaises(RuntimeError):R.read_unit(self.unit)
        self.assertEqual(props.call_count,6)
    def test_transient_group_materialization_is_retried_not_omitted(self):
        starting=dict(self.active,ControlGroup='',MainPID='0',ActiveState='activating',SubState='start')
        with patch.object(R,'properties',side_effect=[starting,self.active]):r=R.read_unit(self.unit)
        self.assertEqual(r['main_pid'],42)
    def test_time_units_preserve_cpu_equivalents(self):
        self.assertEqual(R.cpu_quota('4s'),4)
        self.assertAlmostEqual(R.cpu_quota('100000us'),.1)
        self.assertEqual(R.cpu_quota('100ms'),.1)

class GuardianControls(unittest.TestCase):
    def test_missing_new_review_never_dispatches_or_clears_old_failure(self):
        with tempfile.TemporaryDirectory()as temp,patch.object(T,'REVIEW_ROOT',Path(temp)),patch.object(T.O,'cycle')as cycle:
            self.assertEqual(T.cycle()['phase'],'waiting_exact_recovery_review');cycle.assert_not_called()
    def test_guardian_reuses_original_mlp_restore_and_report_cycle(self):
        with patch.object(T,'manifest',return_value=dict(sources_sha256={'source':'hash'})), \
             patch.object(T.O,'pins'),patch.object(T.O,'cycle',return_value={'phase':'running'})as cycle,patch.object(T.O,'write'):
            self.assertEqual(T.cycle(),{'phase':'running'});cycle.assert_called_once_with()
    def test_recovery_keeps_original_lock_and_old_failures(self):
        text=(HERE/'controller_recovery_v3.py').read_text()
        self.assertIn("ROOT/'control/controller.lock'",text)
        self.assertNotIn('.unlink(',text);self.assertNotIn('request_yield(',text)
        self.assertIn("ROOT/'control_priority_v2/FAILED.json'",text)
    def test_pending_restart_inventory_does_not_reduce_total_caps(self):
        r=R.configured_restart('synrfp-restart.service',dict(LoadState='loaded',ActiveState='activating',SubState='auto-restart',
            MainPID='0',ControlGroup='',CPUQuotaPerSecUSec='2s',MemoryMax=str(4*1024**3)))
        self.assertEqual(r['cpu_equivalents'],2);self.assertEqual(r['memory_max_bytes'],4*1024**3)

class RestoreControls(unittest.TestCase):
    def setUp(self):
        p=patch.object(U.G,'sealed_execution',return_value=None);p.start();self.addCleanup(p.stop)
    def test_restore_replays_verified_execution_after41_was_enabled(self):
        with patch.object(U.G,'sealed_execution',return_value={'phase':'ready'}),patch.object(U.G,'manifest')as manifest, \
             patch.object(U.F,'morgan_report_complete',return_value=True),patch.object(U.F,'report_complete',return_value=True), \
             patch.object(U.O.C,'props',return_value=dict(ActiveState='inactive',MainPID='0')), \
             patch.object(U.O,'snapshot',return_value=dict(units=[])),patch.object(U.F,'restore41',return_value=dict(phase='complete')):
            self.assertEqual(U.cycle()['phase'],'complete');manifest.assert_not_called()
    def test_41_waits_for_full_Morgan_report(self):
        with patch.object(U.G,'manifest',return_value=True),patch.object(U.F,'morgan_report_complete',return_value=False), \
             patch.object(U.F,'restore41')as restore:
            self.assertEqual(U.cycle()['phase'],'waiting_Morgan_full_report');restore.assert_not_called()
    def test_41_waits_for_full_Path_report(self):
        with patch.object(U.G,'manifest',return_value=True),patch.object(U.F,'morgan_report_complete',return_value=True), \
             patch.object(U.F,'report_complete',return_value=False),patch.object(U.F,'restore41')as restore:
            self.assertEqual(U.cycle()['phase'],'waiting_Path_full_report');restore.assert_not_called()
    def test_41_waits_for_recovered_guardian_process_exit(self):
        with patch.object(U.G,'manifest',return_value=True),patch.object(U.F,'morgan_report_complete',return_value=True), \
             patch.object(U.F,'report_complete',return_value=True),patch.object(U.O.C,'props',return_value=dict(ActiveState='inactive',MainPID='9')), \
             patch.object(U.F,'restore41')as restore:
            self.assertEqual(U.cycle()['phase'],'waiting_recovery_guardian_exit');restore.assert_not_called()
    def test_41_waits_for_any_new_Path_report_service(self):
        with patch.object(U.G,'manifest',return_value=True),patch.object(U.F,'morgan_report_complete',return_value=True), \
             patch.object(U.F,'report_complete',return_value=True),patch.object(U.O.C,'props',return_value=dict(ActiveState='inactive',MainPID='0')), \
             patch.object(U.O,'snapshot',return_value=dict(units=[dict(unit='synrfp-preset44-report-v4.service')])),patch.object(U.F,'restore41')as restore:
            self.assertEqual(U.cycle()['phase'],'waiting_all_Path_science_report_and_guardian_exit');restore.assert_not_called()

if __name__=='__main__':unittest.main()
