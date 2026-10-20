import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


P=load('publication46_v3_policy_test',HERE/'publisher_runtime_v3.py')
C=load('publication46_v3_controller_test',HERE/'controller_runtime_v3.py')


class ReportRepairPublicationControls(unittest.TestCase):
    def test_new_namespace_preserves_old_science_and_transaction_functions(self):
        self.assertEqual(P.ROOT.name,'corrected_results_publication_v3')
        self.assertEqual(P.LEGACY_ROOT.name,'corrected_results_publication_v2')
        self.assertEqual(P.MUTEX_ROOT.name,'corrected_results_publication_v1')
        self.assertEqual(Path(P.morgan_gate.__code__.co_filename).name,'publisher.py')
        self.assertEqual(Path(P.ORIGINAL_PATH_GATE.__code__.co_filename).name,'publisher.py')
        self.assertEqual(Path(P.P.commit_and_push.__code__.co_filename).name,'publisher.py')
        self.assertTrue(set(P.FILES)<=set(P.P.OWN))

    def test_missing_recovery_source_cannot_prepare_policy(self):
        with tempfile.TemporaryDirectory()as tmp:
            repo=Path(tmp)
            with patch.object(P.P,'REPO',repo):
                with self.assertRaises(ValueError):P.recovery_sources()
                folder=repo/'Experiments/44_task_preset_comparison';folder.mkdir(parents=True)
                for name in P.RECOVERY_FILES:(folder/name).write_text('bounded report repair\n')
                self.assertEqual(len(P.recovery_sources()),2)

    def test_exact_recovery_catalog_is_required_in_native_path_closure(self):
        with tempfile.TemporaryDirectory()as tmp:
            repo=Path(tmp)/'repo';root=Path(tmp)/'pub';native=Path(tmp)/'44';sources={};authority={}
            for name in P.RECOVERY_FILES:
                relative='Experiments/44_task_preset_comparison/'+name;file=repo/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('verified report adapter\n')
                sources[relative]=P.P.sha(file);authority[str(file.resolve())]=P.P.sha(file)
            P.P.write(root/'config.json',dict(required_path_recovery_sources_sha256=sources))
            P.P.write(native/'COMPLETE.json',dict(outputs_sha256=authority))
            with patch.object(P,'ROOT',root),patch.object(P.P,'ROOT44',native),patch.object(P.P,'REPO',repo),patch.object(P,'ORIGINAL_PATH_GATE',return_value={'phase':'complete'})as original:
                self.assertEqual(P.path_gate(),{'phase':'complete'});original.assert_called_once()
                authority.pop(next(iter(authority)));P.P.write(native/'COMPLETE.json',dict(outputs_sha256=authority))
                with self.assertRaisesRegex(ValueError,'omits the reviewed report recovery'):P.path_gate()

    def test_partition_export_path_is_preserved_under_new_version(self):
        with patch.object(P,'ORIGINAL_MAIN_PATHS',return_value=['partition.json'])as original:
            value=P.main_paths('path',dict(publication_runtime_version=3,marker='retained'))
        self.assertEqual(value,['partition.json'])
        self.assertEqual(original.call_args.args[1],dict(publication_runtime_version=2,marker='retained'))

    def test_fresh_config_includes_recovery_sources_and_copies_no_old_changes(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)/'new';old=Path(tmp)/'old';P.P.write(old/'config.json',dict(retained=True));before=(old/'config.json').read_bytes()
            sources={'Experiments/44_task_preset_comparison/'+name:'a'*64 for name in P.RECOVERY_FILES}
            base=dict(component_sources={'path':list(sources)},old_completed_publication_proofs_sha256={})
            with patch.object(P,'ROOT',root),patch.object(P,'LEGACY_ROOT',old),patch.object(P,'recovery_sources',return_value=sources),patch.object(P.V,'prepare',return_value=base),patch.object(P.V,'adopt_completed_legacy',return_value={}):
                value=P.prepare()
            self.assertEqual(value['publication_runtime_version'],3);self.assertEqual(value['required_path_recovery_sources_sha256'],sources)
            self.assertEqual((old/'config.json').read_bytes(),before)
            self.assertEqual(P.P.read(root/'config.json'),value)

    def test_cpu_only_job_and_governor_only_handoff(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            for name in ('config.json','CODE_READY.json','ROOT_REVIEW.json'):P.P.write(root/name,{})
            with patch.object(C,'ROOT',root),patch.object(C.O,'ROOT',root):job=C.spec('path',{'sources_sha256':{}})
        self.assertTrue(job['id'].startswith('publication46v3-'));self.assertTrue(job['arguments'][0].endswith('publisher_runtime_v3.py'))
        self.assertEqual((job['cpu_equivalents'],job['memory_gib']),(2,1));self.assertEqual(job['environment']['CUDA_VISIBLE_DEVICES'],'')
        with patch.object(C,'review',return_value={}),patch.object(C,'old_publication_idle',return_value=False),patch.object(C.O,'cycle')as cycle,patch.object(C.subprocess,'run')as mutate:
            self.assertIn('waiting_previous',C.cycle()['phase']);cycle.assert_not_called();mutate.assert_not_called()

    def test_path_can_finish_while_morgan_remains_pending(self):
        state=dict(phase='waiting_or_publishing_independent_components',completed=dict(path=True,morgan=False))
        with patch.object(C,'review',return_value={}),patch.object(C,'old_publication_idle',return_value=True),patch.object(C.O,'cycle',return_value=state):
            self.assertEqual(C.cycle(),state)


if __name__=='__main__':unittest.main()
