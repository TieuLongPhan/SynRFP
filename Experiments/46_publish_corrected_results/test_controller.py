import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('publisher46_controller_tests',Path(__file__).with_name('controller.py'))
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

class IndependentWatcherControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.saved=(C.ROOT,C.CONTROL,C.P.ROOT43,C.P.ROOT44)
        C.ROOT=self.root/'publisher';C.CONTROL=C.ROOT/'control';C.P.ROOT43=self.root/'43';C.P.ROOT44=self.root/'44';C.ROOT.mkdir()
        C.P.write(C.ROOT/'config.json',dict(policy='reviewed'));C.P.write(C.ROOT/'CODE_READY.json',dict(phase='complete'));C.P.write(C.ROOT/'ROOT_REVIEW.json',dict(phase='reviewed'));self.review={'sources_sha256':{str(Path(__file__)):C.P.sha(__file__)}}
    def tearDown(self):C.ROOT,C.CONTROL,C.P.ROOT43,C.P.ROOT44=self.saved;self.tmp.cleanup()
    def proof(self,component):C.P.write(C.ROOT/component/'PUBLISHED.json',dict(phase='complete',component=component,branch=C.P.BRANCH,weights_published=False,other_component_waited_for=False,commit='a'*40))
    def inactive(self,*args):return dict(ActiveState='inactive',MainPID='0')
    def test_one_completed_component_never_stops_watcher_for_other(self):
        self.proof('path')
        with patch.object(C,'review',return_value=self.review),patch.object(C.C,'props',side_effect=self.inactive),patch.object(C.C,'attest_service'),patch.object(C.C,'ensure')as launch:
            r=C.cycle();self.assertNotEqual(r['phase'],'complete');self.assertTrue(r['completed']['path']);self.assertFalse(r['completed']['morgan']);launch.assert_not_called()
    def test_path_triggers_without_morgan_completion(self):
        C.P.write(C.P.ROOT44/'COMPLETE.json',dict(phase='complete'))
        with patch.object(C,'review',return_value=self.review),patch.object(C.C,'props',side_effect=self.inactive),patch.object(C.C,'attest_service'),patch.object(C.C,'ensure',return_value={'phase':'resumed_or_launched'})as launch:
            r=C.cycle();self.assertEqual(launch.call_args.args[0]['id'],'publication46-path');self.assertNotEqual(r['phase'],'complete')
    def test_morgan_triggers_without_path_completion(self):
        C.P.write(C.P.ROOT43/'COMPLETE.json',dict(phase='complete'))
        with patch.object(C,'review',return_value=self.review),patch.object(C.C,'props',side_effect=self.inactive),patch.object(C.C,'attest_service'),patch.object(C.C,'ensure',return_value={'phase':'resumed_or_launched'})as launch:
            C.cycle();self.assertEqual(launch.call_args.args[0]['id'],'publication46-morgan')
    def test_latched_component_does_not_starve_other(self):
        for root in (C.P.ROOT43,C.P.ROOT44):C.P.write(root/'COMPLETE.json',dict(phase='complete'))
        with patch.object(C,'review',return_value=self.review),patch.object(C.C,'props',side_effect=self.inactive),patch.object(C.C,'attest_service'),patch.object(C.C,'ensure',side_effect=[{'phase':'latched'},{'phase':'resumed_or_launched'}])as launch:
            C.cycle();self.assertEqual(launch.call_count,2)
    def test_verified_governor_pause_releases_only_future_legacy_reserve(self):
        value=dict(physical_cores=32,memory_total_bytes=128*C.GIB,memory_available_bytes=100*C.GIB,cpu_cap_sum=20,memory_cap_sum_bytes=70*C.GIB,disk_free_bytes=60*C.GIB,units=[])
        with patch.object(C.PAUSE,'paused41',return_value=True),patch.object(C.PRIORITY,'priority_context')as legacy:
            self.assertEqual(C.admission(value,2,C.GIB),[]);legacy.assert_not_called()
        with patch.object(C.PAUSE,'paused41',return_value=False),patch.object(C.PRIORITY,'priority_context',return_value=dict(active=True,pending_cpu=2,pending_memory_bytes=24*C.GIB)):
            self.assertIn('clustering_priority_memory_headroom',C.admission(value,2,C.GIB))
        with patch.object(C.PAUSE,'paused41',side_effect=ValueError('mutated proof')):
            with self.assertRaisesRegex(ValueError,'mutated proof'):C.admission(value,2,C.GIB)

    def test_publisher_limits_cuda_hidden_and_no_lease(self):
        j=C.spec('path',self.review);self.assertEqual((j['cpu_equivalents'],j['memory_gib']),(2,1));self.assertEqual(j['environment']['CUDA_VISIBLE_DEVICES'],'');self.assertNotIn('gpu',j);self.assertNotIn('lock',str(j['arguments']))
    def test_both_publication_proofs_close_watcher(self):
        self.proof('path');self.proof('morgan')
        with patch.object(C,'review',return_value=self.review),patch.object(C.C,'ensure')as launch:
            r=C.cycle();self.assertEqual(r['phase'],'complete');self.assertEqual(len(r['outputs_sha256']),2);launch.assert_not_called()
    def test_missing_review_never_touches_git_or_services(self):
        with patch.object(C,'review',return_value=None),patch.object(C.C,'ensure')as launch,patch.object(C.P,'git')as git:
            self.assertIn('waiting',C.cycle()['phase']);launch.assert_not_called();git.assert_not_called()

if __name__=='__main__':unittest.main()
