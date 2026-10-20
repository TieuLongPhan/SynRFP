import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('parallel44_test_target',HERE/'controller_parallel_v2.py')
T=importlib.util.module_from_spec(s);s.loader.exec_module(T)

class ParallelControls(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        for obj,key,value in ((T,'ROOT',self.root),(T,'CONTROL',self.root/'control_parallel_v2'),(T,'ADMITTED',{'sources_sha256':{}}),
                              (T.O,'ROOT',self.root)):
            p=patch.object(obj,key,value);p.start();self.addCleanup(p.stop)
        T.O.write(self.root/'clustering/plan.json',dict(tasks=[]))
        T.O.write(T.CONTROL/'execution.json',dict(reviewed=True))
        self.execution=dict(plan_sha256='a'*64,sources_sha256={})
    def test_exact_five_cpu_worker_and_unchanged_identity(self):
        with patch.object(T.O,'dispatched',return_value=[]):job=T.worker_spec(2,self.execution)
        self.assertEqual((job['cpu_equivalents'],job['memory_gib'],job['cores']),(5,12,'18,19,26,27,28'))
        self.assertEqual(job['run_id'],'c0-g0002')
        self.assertEqual(job['arguments'][job['arguments'].index('--threads')+1],'5')
        self.assertEqual(job['environment']['CUDA_VISIBLE_DEVICES'],'')
        self.assertNotIn('gpu',job)
    def test_preliminary_admission_accounts_five_actual_cpus(self):
        with patch.object(T,'ORIGINAL_ADMISSION',return_value=[]) as admission:
            T.admission({},1,12*T.O.GIB)
            admission.assert_called_once_with({},5,12*T.O.GIB)
    def test_other_profiles_are_not_relabelled(self):
        with patch.object(T,'ORIGINAL_ADMISSION',return_value=[]) as admission:
            T.admission({},1,8*T.O.GIB)
            admission.assert_called_once_with({},1,8*T.O.GIB)
    def test_old_live_worker_prevents_new_generation(self):
        with patch.object(T.O,'dispatched',return_value=[{'stage':'clustering','unit':'old'}]), \
             patch.object(T.O.C,'props',return_value=dict(ActiveState='active',MainPID='99')):
            with self.assertRaises(T.O.C.IntegrityError):T.worker_spec(2,self.execution)
    def test_old_inactive_nonzero_pid_prevents_new_generation(self):
        with patch.object(T.O,'dispatched',return_value=[{'stage':'clustering','unit':'old'}]), \
             patch.object(T.O.C,'props',return_value=dict(ActiveState='inactive',MainPID='99')):
            with self.assertRaises(T.O.C.IntegrityError):T.worker_spec(2,self.execution)
    def test_old_worker_requires_audited_terminal(self):
        with patch.object(T.O,'dispatched',return_value=[{'stage':'clustering','unit':'old'}]), \
             patch.object(T.O.C,'props',return_value=dict(ActiveState='inactive',MainPID='0')),patch.object(T.O,'terminal',return_value=None):
            with self.assertRaises(T.O.C.IntegrityError):T.worker_spec(2,self.execution)
    def test_audited_yield_allows_fresh_five_thread_invocation(self):
        with patch.object(T.O,'dispatched',return_value=[{'stage':'clustering','unit':'old'}]), \
             patch.object(T.O.C,'props',return_value=dict(ActiveState='inactive',MainPID='0')),patch.object(T.O,'terminal',return_value='yielded'):
            self.assertEqual(T.worker_spec(2,self.execution)['cpu_equivalents'],5)
    def test_original_job_specification_is_not_mutated(self):
        old=T.ORIGINAL_WORKER_SPEC(1,self.execution)
        with patch.object(T.O,'dispatched',return_value=[]):new=T.worker_spec(2,self.execution)
        self.assertEqual(old['cpu_equivalents'],1)
        self.assertEqual(old['arguments'][old['arguments'].index('--threads')+1],'1')
        self.assertNotEqual(old['run_id'],new['run_id'])
    def test_adapter_keeps_original_lock_and_never_signals_science(self):
        text=(HERE/'controller_parallel_v2.py').read_text()
        self.assertIn("ROOT/'control/controller.lock'",text)
        self.assertNotIn('request_yield(',text)
        self.assertNotIn('lease_available(',text)
        self.assertIn('result=O.cycle()',text)
    def test_missing_parallel_review_does_not_dispatch(self):
        with patch.object(T,'REVIEW_ROOT',self.root/'absent'),patch.object(T.O,'cycle') as cycle:
            self.assertEqual(T.cycle()['phase'],'waiting_reviewed_parallel_continuation')
            cycle.assert_not_called()

if __name__=='__main__':unittest.main()
