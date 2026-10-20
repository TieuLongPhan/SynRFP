import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def load(name,path):
    q=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(q);q.loader.exec_module(m);return m

V=load('publisher46_v2_test',HERE/'publisher_runtime_v2.py')
C=load('controller46_v2_test',HERE/'controller_runtime_v2.py')

class VersionedPublicationControls(unittest.TestCase):
    def test_isolated_policy_does_not_modify_old_root_or_scientific_gates(self):
        self.assertNotEqual(V.ROOT,V.LEGACY_ROOT);self.assertEqual(V.ROOT.name,'corrected_results_publication_v2')
        self.assertIs(V.path_gate,V.P.path_gate);self.assertIs(V.morgan_gate,V.P.morgan_gate)
        self.assertTrue(all(n in V.P.OWN for n in V.FILES))
        self.assertEqual(Path(V.P.morgan_gate.__code__.co_filename).name,'publisher.py')
    def test_all_eight_future_v3_sources_are_required_and_allowlisted(self):
        with tempfile.TemporaryDirectory()as tmp:
            repo=Path(tmp);folder=repo/'Experiments/44_task_preset_comparison';folder.mkdir(parents=True)
            with patch.object(V.P,'REPO',repo),patch.object(V.P,'EX44',folder/'publication_metadata/task_preset_comparison_v1'):
                with self.assertRaises(ValueError):V.require_v3_sources()
                for name in V.V3_FILES:(folder/name).write_text('reviewed five-lane source\n')
                pins=V.require_v3_sources();self.assertEqual(len(pins),8)
                config=dict(component_sources={'path':sorted(pins),'morgan':[]},publisher_paths=[])
                paths=V.P.main_paths('path',config)
                self.assertTrue(set(pins)<=set(paths));self.assertTrue(any(p.endswith('report_parallel_v3.py')for p in paths))
    def test_partition_provenance_requires_native_authority_before_copy(self):
        with tempfile.TemporaryDirectory()as tmp:
            repo=Path(tmp)/'repo';root=Path(tmp)/'44';out=repo/'Experiments/44_task_preset_comparison/publication_metadata/task_preset_comparison_v1'
            catalog={}
            for name in V.PROVENANCE_FILES:
                p=root/'parallel_v3'/name;V.P.write(p,dict(partition='five disjoint lanes'));catalog[str(p)]=V.P.sha(p)
            V.P.write(root/'COMPLETE.json',dict(outputs_sha256=catalog))
            config=dict(component_sources={'path':[]},publisher_paths=[],publication_runtime_version=2)
            with patch.object(V.P,'ROOT44',root),patch.object(V.P,'REPO',repo),patch.object(V.P,'EX44',out):
                paths=V.main_paths('path',config);self.assertTrue(all(str((out/'parallel_v3'/n).relative_to(repo))in paths for n in V.PROVENANCE_FILES))
                (root/'parallel_v3/plan.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'provenance changed'):V.main_paths('path',config)

    def test_old_publication_receipts_remain_byte_identical_when_adopted(self):
        with tempfile.TemporaryDirectory()as tmp:
            old=Path(tmp)/'old';new=Path(tmp)/'new';source=old/'path/PUBLISHED.json'
            V.P.write(source,dict(phase='complete',component='path',branch=V.P.BRANCH,weights_published=False,commit='a'*40))
            before=source.read_bytes()
            with patch.object(V,'LEGACY_ROOT',old),patch.object(V,'ROOT',new):pins=V.adopt_completed_legacy()
            self.assertEqual(source.read_bytes(),before);self.assertEqual((new/'path/PUBLISHED.json').read_bytes(),before);self.assertIn(str(source.resolve()),pins)
    def test_v2_job_uses_new_cli_root_and_cpu_only_limits(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);V.P.write(root/'config.json',{});V.P.write(root/'CODE_READY.json',{});V.P.write(root/'ROOT_REVIEW.json',{})
            with patch.object(C,'ROOT',root),patch.object(C.O,'ROOT',root):
                j=C.spec('path',{'sources_sha256':{}})
            self.assertTrue(j['id'].startswith('publication46v2-'));self.assertTrue(j['arguments'][0].endswith('publisher_runtime_v2.py'))
            self.assertEqual((j['cpu_equivalents'],j['memory_gib']),(2,1));self.assertEqual(j['environment']['CUDA_VISIBLE_DEVICES'],'')
    def test_new_governor_waits_for_old_governor_without_stopping_it(self):
        with patch.object(C,'review',return_value={}),patch.object(C,'old_publication_idle',return_value=False),patch.object(C.O,'cycle')as work,patch.object(C.subprocess,'run')as mutate:
            self.assertIn('waiting_original46',C.cycle()['phase']);work.assert_not_called();mutate.assert_not_called()
    def test_independent_trigger_state_machine_is_reused(self):
        with patch.object(C,'review',return_value={}),patch.object(C,'old_publication_idle',return_value=True),patch.object(C.O,'cycle',return_value={'phase':'waiting_or_publishing_independent_components','completed':{'path':True,'morgan':False}}):
            r=C.cycle();self.assertNotEqual(r['phase'],'complete');self.assertTrue(r['completed']['path']);self.assertFalse(r['completed']['morgan'])

if __name__=='__main__':unittest.main()
