"""CPU-only recovery, fatal visibility and honest partial-output controls."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cpu_recovery_control_tests', HERE/'controller_recovery_v3.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
spec = importlib.util.spec_from_file_location('partial_refresh_tests', HERE/'report_refresh_v3.py')
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


class ScientificVisibility(unittest.TestCase):
    def test_known_replay_failure_is_immediately_fatal(self):
        for message in ('ValueError: Saved center prediction replay differs',
                        'ValueError: True independent-center regret exceeds bound',
                        'RuntimeError: fitted/predict mismatch', 'AssertionError: invalid permutation'):
            self.assertTrue(C.scientific_failure(message))
        self.assertFalse(C.scientific_failure('service exited due to SIGTERM'))

    def test_all_latched_workers_cannot_look_running(self):
        decisions = {'worker'+str(i): dict(phase='latched') for i in range(8)}
        self.assertEqual(len(C.blocked(decisions)), 8)
        with patch.object(C.B,'state',return_value='inactive'):
            value = C.blocked_status(decisions, {})
        self.assertEqual(value['phase'],'blocked_by_scientific_failure')
        self.assertFalse(value['automatic_bypass'])

    def test_one_failure_remains_visible_while_other_workers_run(self):
        decisions = dict(worker0=dict(phase='latched'),worker1=dict(phase='running'))
        self.assertEqual(set(C.blocked(decisions)),{'worker0'})

    def test_known_failure_latches_before_ensure_can_retry(self):
        job=dict(id='recovery-worker0',unit='owned',work_done={})
        with patch.object(C.C,'props',return_value=dict(ActiveState='failed')), patch.object(C.C,'attest_service'), \
             patch.object(C.C,'gate',return_value=False), patch.object(C.C,'failure_logs',return_value='ValueError: Saved center prediction replay differs'), \
             patch.object(C.C,'latch') as latch, patch.object(C.N,'safe_ensure',return_value=dict(phase='latched')) as ensure:
            value=C.ensure(job)
        latch.assert_called_once()
        self.assertEqual(latch.call_args.args[1],'scientific_validation_failure_no_retry')
        ensure.assert_called_once()
        self.assertEqual(value['phase'],'latched')

    def test_latched_siblings_are_frozen_without_gpu_action(self):
        specs={'worker1':dict(id='x',unit='cpu-owned')}
        with patch.object(C.B,'state',return_value='active'),patch.object(C.C,'props',return_value={}), \
             patch.object(C.C,'attest_service'),patch.object(C.C,'freeze_or_thaw') as freeze:
            result=C.blocked_status(dict(worker0=dict(phase='latched')),specs)
        freeze.assert_called_once_with('freeze','cpu-owned')
        self.assertEqual(result['frozen_recovery_workers'],['cpu-owned'])


class Scope(unittest.TestCase):
    def test_no_gpu_job_or_old_failed_job_can_be_admitted(self):
        jobs=C.jobs(dict(sources_sha256={}))
        self.assertTrue(all('gpu' not in j for j in jobs.values()))
        self.assertTrue(all(j['environment']['CUDA_VISIBLE_DEVICES']=='' for j in jobs.values()))
        self.assertTrue(all('regression' not in j['id'] and 'parallel' not in j['id'] for j in jobs.values()))
        self.assertTrue(all(j['unit'].endswith('-v3') for j in jobs.values()))

    def test_parallel_and_refresh_caps_fit_75percent_limit(self):
        jobs=C.jobs(dict(sources_sha256={}))
        models=[jobs['worker'+str(i)]for i in range(8)]
        self.assertEqual(sum(j['memory_gib']for j in models)+jobs['refresh']['memory_gib']+1,89)
        self.assertAlmostEqual(sum(j['cpu_equivalents']for j in models)+jobs['refresh']['cpu_equivalents']+.1,18.1)
        self.assertEqual([j['cores']for j in models],[f'{8+2*i}-{9+2*i}'for i in range(8)])

    def test_foreign_or_untested_readiness_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ready.json'
            C.B.write(path,dict(phase='ready',tests_passed=True,sources_sha256={}))
            with self.assertRaisesRegex(ValueError,'exact tested'):
                C.checked_ready(path,C.OWN)

    def test_previous_worker_must_release(self):
        with patch.object(C.B,'state',return_value='active'):
            self.assertFalse(C.old_workers_released())
        with patch.object(C.B,'state',return_value='failed'):
            self.assertTrue(C.old_workers_released())


class PartialTruth(unittest.TestCase):
    def test_six_ready_tasks_are_visible_while_clustering_missing(self):
        rows={t:[{}]*n for t,n in R.R.EXPECTED.items()if t!='clustering'}
        rows['clustering']=[]
        result=R.build_status(rows)
        self.assertEqual(result['total_completed'],6798)
        self.assertEqual(result['phase'],'partial')
        self.assertFalse(result['overall_complete'])
        self.assertTrue(all(result['tasks'][t]['complete']for t in R.R.EXPECTED if t!='clustering'))

    def test_partial_cluster_cells_do_not_imply_final_completion(self):
        rows={t:[{}]*n for t,n in R.R.EXPECTED.items()if t!='clustering'}
        rows['clustering']=[{}]*1248
        result=R.build_status(rows)
        self.assertEqual(result['total_completed'],8046)
        self.assertFalse(result['overall_complete'])
        self.assertFalse(result['tasks']['clustering']['complete'])

    def test_even_all_raw_cells_need_native_final_receipt(self):
        result=R.build_status({t:[{}]*n for t,n in R.R.EXPECTED.items()})
        self.assertEqual(result['total_completed'],8238)
        self.assertTrue(result['native_final_receipt_required'])
        self.assertFalse(result['overall_complete'])

    def test_extra_cell_is_refused(self):
        with self.assertRaisesRegex(ValueError,'universe'):
            R.build_status(dict(classification=[{}]*4201))


if __name__=='__main__':
    unittest.main()
