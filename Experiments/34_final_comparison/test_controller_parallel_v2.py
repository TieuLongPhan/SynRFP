"""Exercise native ownership, bounded scheduling and narrow transient recovery."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('parallel_controller_tests', Path(__file__).with_name('controller_parallel_v2.py'))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class Recovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.job = dict(id='owned', unit='owned.service', sources_sha256={}, work_done={}, cpu_equivalents=2, memory_gib=10)
        self.error = subprocess.CalledProcessError(5, ['systemctl', '--user', 'start', 'owned.service'])

    def attempt(self, *, error=None, state=None, latch=False, admission=None, gpu_free=True):
        if latch:
            p = self.base / 'helper/latches/owned.json'
            p.parent.mkdir(parents=True)
            p.write_text('{}')
        stack = []
        def enter(value):
            entered = value.__enter__()
            stack.append(value)
            return entered
        try:
            enter(patch.object(R, 'V2', self.base / 'v2'))
            enter(patch.object(R.C, 'ROOT', self.base / 'helper'))
            ensure = enter(patch.object(R.C, 'ensure', side_effect=[error or self.error, {'phase': 'running'}]))
            enter(patch.object(R.C, 'props', return_value=state or dict(LoadState='not-found', ActiveState='inactive')))
            enter(patch.object(R.C, 'attest_service'))
            enter(patch.object(R.C, 'gate', return_value=False))
            enter(patch.object(R.B, 'snapshot', return_value={}))
            enter(patch.object(R.B, 'admission', return_value=admission or []))
            enter(patch.object(R.B, 'lease_available', return_value=gpu_free))
            launch = enter(patch.object(R.B, 'launch', return_value={'phase': 'launched'}))
            return R.safe_ensure(self.job, Mock()), ensure, launch
        finally:
            for value in reversed(stack):
                value.__exit__(None, None, None)

    def test_only_vanished_transient_is_recreated_and_reattested(self):
        result, ensure, launch = self.attempt()
        self.assertEqual(result['phase'], 'running')
        self.assertEqual(ensure.call_count, 2)
        launch.assert_called_once_with(self.job)
        self.assertTrue((self.base / 'v2/last_transient_recovery.json').exists())

    def test_other_exit_codes_are_not_recovered(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.attempt(error=subprocess.CalledProcessError(1, self.error.cmd))

    def test_other_unit_names_are_not_recovered(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.attempt(error=subprocess.CalledProcessError(5, ['systemctl', '--user', 'start', 'foreign.service']))

    def test_still_loaded_or_active_unit_is_not_recreated(self):
        for state in (dict(LoadState='loaded', ActiveState='inactive'), dict(LoadState='not-found', ActiveState='active')):
            with self.assertRaises(subprocess.CalledProcessError):
                self.attempt(state=state)

    def test_scientific_latch_is_never_bypassed(self):
        with self.assertRaisesRegex(RuntimeError, 'scientific latch'):
            self.attempt(latch=True)

    def test_resource_guards_are_rechecked_before_recreation(self):
        value, ensure, launch = self.attempt(admission=['memory_budget'])
        self.assertEqual(value['phase'], 'waiting_resources')
        launch.assert_not_called()
        self.assertEqual(ensure.call_count, 1)

    def test_busy_physical_gpu_is_not_recreated(self):
        self.job['gpu'] = 1
        value, ensure, launch = self.attempt(gpu_free=False)
        self.assertEqual(value['phase'], 'waiting_physical_gpu_lease')
        launch.assert_not_called()


class Bounds(unittest.TestCase):
    def test_eight_workers_fit_exact_parent_budget(self):
        with patch.object(R.C, 'read', return_value=dict(sources_sha256={})):
            jobs = R.jobs(dict(sources_sha256={}))
        workers = [jobs['parallel' + str(i)] for i in range(8)]
        self.assertEqual(sum(j['memory_gib'] for j in workers) + jobs['regression']['memory_gib'] + 1, 93)
        self.assertAlmostEqual(sum(j['cpu_equivalents'] for j in workers) + jobs['regression']['cpu_equivalents'] + .1, 18.1)
        self.assertTrue(all(j['memory_gib'] == 10 and 'gpu' not in j for j in workers))
        self.assertEqual([j['cores'] for j in workers], [f'{8+2*i}-{9+2*i}' for i in range(8)])
        self.assertEqual(jobs['regression']['unit'], 'synrfp-final-comparison-regression-v1')
        self.assertEqual(jobs['clustering_pilot']['unit'], 'synrfp-final-comparison-clustering-pilot-v1')
        self.assertEqual(jobs['regression']['gpu'], 1)
        self.assertNotIn('clustering0', jobs)

    def test_original_worker_must_release_before_new_admission(self):
        with patch.object(R.B, 'state', side_effect=lambda unit: 'active' if 'clustering2-' in unit else 'inactive'):
            self.assertFalse(R.native_workers_inactive())
        with patch.object(R.B, 'state', return_value='inactive'):
            self.assertTrue(R.native_workers_inactive())

    def test_pilot_memory_evidence_has_headroom_and_cannot_be_faked(self):
        self.assertTrue(R.pilot_memory_ok(dict(peak_rss_gib=8.9)))
        for bad in (None, True, 0, -1, 9.0, 9.1, float('nan'), float('inf'), '5'):
            self.assertFalse(R.pilot_memory_ok(dict(peak_rss_gib=bad)))

    def test_parallel_readiness_requires_exact_tested_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(R, 'ROOT', Path(directory)):
            p = Path(directory) / 'clustering/parallel_v2/CODE_READY.json'
            R.B.write(p, dict(phase='ready', tests_passed=True, sources_sha256={}))
            with self.assertRaisesRegex(ValueError, 'meaningful'):
                R.parallel_ready()
            R.B.write(p, dict(phase='ready', tests_passed=2, sources_sha256={'/foreign': 'bad'}))
            with self.assertRaisesRegex(ValueError, 'exactly'):
                R.parallel_ready()


if __name__ == '__main__':
    unittest.main()
