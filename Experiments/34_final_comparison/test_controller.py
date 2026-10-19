import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('final_comparison_controller_tests', Path(__file__).with_name('controller.py'))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class Admission(unittest.TestCase):
    def test_no_sources_are_imported_before_tested_ready(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(C, 'ROOT', Path(temp)):
            self.assertIsNone(C.ready('regression'))
            C.B.write(Path(temp) / 'regression/CODE_READY.json', dict(phase='updating'))
            self.assertIsNone(C.ready('regression'))

    def test_readiness_cannot_bind_foreign_sources_or_fake_test_count(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(C, 'ROOT', Path(temp)):
            p = Path(temp) / 'regression/CODE_READY.json'
            C.B.write(p, dict(phase='ready', tests_passed=True, sources_sha256={}))
            with self.assertRaisesRegex(ValueError, 'tested readiness'):
                C.ready('regression')
            C.B.write(p, dict(phase='ready', tests_passed=3, sources_sha256={'/foreign': 'bad'}))
            with self.assertRaisesRegex(ValueError, 'unexpected source'):
                C.ready('regression')

    def test_exact_worker_limits_and_gpu_dispatch(self):
        jobs = C.jobs(dict(sources_sha256={}))
        active_keys = ['regression', 'report', 'clustering0', 'clustering1', 'clustering2', 'clustering3']
        self.assertEqual(sum(jobs[k]['memory_gib'] for k in active_keys) + 1, 85)
        self.assertLessEqual(sum(jobs[k]['cpu_equivalents'] for k in active_keys) + .1, 24)
        self.assertEqual(jobs['regression']['gpu'], 1)
        self.assertTrue(all('gpu' not in jobs['clustering'+str(i)] for i in range(4)))
        self.assertEqual(jobs['regression']['work_done']['equals']['completed'], 174)

    def test_receipt_alone_does_not_release_active_producer(self):
        with patch.object(C.C, 'gate', return_value=True), patch.object(C.B, 'state', return_value='active'):
            self.assertFalse(C.finished(dict(work_done={}, unit='owned')))
        with patch.object(C.C, 'gate', return_value=False), patch.object(C.B, 'state', return_value='inactive'):
            self.assertFalse(C.finished(dict(work_done={}, unit='owned')))


if __name__ == '__main__':
    unittest.main()
