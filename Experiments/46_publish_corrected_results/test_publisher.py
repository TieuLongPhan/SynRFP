import copy
import hashlib
import importlib.util
import itertools
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('publisher46_tests',Path(__file__).with_name('publisher.py'))
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)

class ScientificPolicyControls(unittest.TestCase):
    def test_exact_independent_component_headers(self):
        a=dict(schema='morgan43-full-correction-complete-v1',phase='complete',tokenizer_profile='synrfp-ecfp-count-v1',all_keyed_closures_verified=True,corrected_mlp_cells=1680,corrected_clustering_cells=1440,combined_classification_cells=5040,combined_clustering_cells=4320)
        P.check_morgan_header(a)
        for key,value in [('corrected_mlp_cells',1),('tokenizer_profile','oldMorgan'),('combined_clustering_cells',4319)]:
            with self.assertRaises(ValueError):P.check_morgan_header(dict(a,**{key:value}))
        b=dict(schema='synrfp-task-preset-comparison-complete-v1',phase='complete',task='clustering',candidate=P.PATH_CONFIG,candidate_cells=240,baseline_cells=1200,primary_cells=1440,cells_with_shared_control=1680,initialization_seeds=10,all_keyed_cells_verified=True,all_retained_artifact_hashes_verified=True,classification_deferred=True)
        P.check_path_header(b)
        for key,value in [('task','classification'),('candidate_cells',20),('initialization_seeds',5),('all_retained_artifact_hashes_verified',False)]:
            with self.assertRaises(ValueError):P.check_path_header(dict(b,**{key:value}))

    def test_keyed_crossing_catches_duplicate_and_foreign_cells(self):
        expected=set(itertools.product(('a','b'),('c','d'),range(5)));rows=[dict(dataset=d,config=c,fold=str(f))for d,c,f in expected]
        P.keyed_grid(rows,expected,('dataset','config','fold'))
        with self.assertRaisesRegex(ValueError,'keyed'):P.keyed_grid(rows[:-1]+[rows[0]],expected,('dataset','config','fold'))
        other=copy.deepcopy(rows);other[0]['fold']='9'
        with self.assertRaisesRegex(ValueError,'keyed'):P.keyed_grid(other,expected,('dataset','config','fold'))

    def test_whitelist_rejects_weights_arrays_symlinks_and_large_metadata(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);good=root/'results.csv';good.write_text('mcc\n0.5\n');self.assertEqual(P.safe_path(root,'results.csv'),good)
            for rel in ('models/model.json','features/fp.json','result.npz','weights.pt','../escape.json'):
                with self.assertRaises(ValueError):P.safe_path(root,rel)
            (root/'alias.csv').symlink_to(good)
            with self.assertRaises(ValueError):P.safe_path(root,'alias.csv')
            large=root/'huge.json'
            with large.open('wb')as f:f.truncate(20*1024**2+1)
            with self.assertRaisesRegex(ValueError,'20MiB'):P.safe_path(root,'huge.json')

    def test_scalar_exports_cannot_contain_raw_reactions_or_predictions(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'bad.csv';p.write_text('dataset,rxn\na,C>>O\n')
            with self.assertRaisesRegex(ValueError,'Raw data'):P.csv_rows(p)

    def test_main_snapshot_never_accepts_package_source_and_excludes_deferred_classification(self):
        config=dict(component_sources={'morgan':['synrfp/tokenizers/morgan.py'],'path':[]},publisher_paths=[])
        with self.assertRaisesRegex(ValueError,'package source'):P.main_paths('morgan',config)
        self.assertFalse(any(Path(p).name in ('classification.py','test_classification.py')for p in P.source_files('Experiments/44_task_preset_comparison')))

    def test_minimal_runtime_helper_allowlist_contains_only_small_sources(self):
        paths=list(P.HELPERS)+['Experiments/43_morgan_ecfp_correction/'+n for n in P.PATH_OPERATIONAL43]
        self.assertIn('Experiments/42_knn_factorial/resource85.py',paths)
        self.assertIn('Experiments/20_focused_factorial/per_task_analysis.py',paths)
        for relative in paths:
            p=P.safe_path(P.REPO,relative);self.assertIn(p.suffix,{'.py','.md','.service'});self.assertLess(p.stat().st_size,1024**2)

    def test_path_metadata_uses_actual_frozen_challenger_decision(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);decision=root/'decisions/clustering_path_challenger.json';P.write(decision,dict(phase='frozen'))
            P.write(root/'clustering/plan.json',dict(decision_path=str(decision),decision_sha256=P.sha(decision)))
            with patch.object(P,'ROOT44',root):
                self.assertEqual(P.path_metadata_bindings()[0],(decision,'decision.json'))
                decision.write_text('{}')
                with self.assertRaisesRegex(ValueError,'decision digest'):P.path_metadata_bindings()

    def test_declared_digest_mutation_is_caught_after_cached_validation(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'array.npz';p.write_bytes(b'kept locally');cache=P.HashCache();h=P.sha(p);P.verify_catalog({str(p):h},cache)
            st=p.stat();p.write_bytes(b'changed bytes');os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'digest changed'):P.verify_catalog({str(p):h},cache)

class PrivateGitControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.repo=self.root/'repo';self.repo.mkdir()
        P.git(self.repo,'init','-q','-b',P.BRANCH);P.git(self.repo,'config','user.name','Publication Test');P.git(self.repo,'config','user.email','test@example.invalid')
        (self.repo/'result.csv').write_text('mcc\n0.1\n');(self.repo/'unrelated.txt').write_text('original\n');(self.repo/'synrfp').mkdir();(self.repo/'synrfp/morgan.py').write_text('old sealed source\n')
        P.git(self.repo,'add','--','result.csv','unrelated.txt','synrfp/morgan.py');P.git(self.repo,'commit','-qm','base');self.base=P.git(self.repo,'rev-parse','HEAD').strip()
        self.remote=self.root/'remote.git';P.git(self.root,'init','--bare','-q',str(self.remote));P.git(self.repo,'remote','add','origin',str(self.remote));P.git(self.repo,'push','-q','origin',P.BRANCH)
        self.other=self.root/'other';P.git(self.root,'clone','-q','--branch',P.BRANCH,str(self.remote),str(self.other));P.git(self.other,'config','user.name','Other Test');P.git(self.other,'config','user.email','other@example.invalid')
        (self.repo/'result.csv').write_text('mcc\n0.9\n');(self.repo/'unrelated.txt').write_text('user staged\n');P.git(self.repo,'add','--','unrelated.txt');(self.repo/'untracked.txt').write_text('retain user file\n')
        self.index=self.repo/'.git/index';self.index_before=P.sha(self.index);self.hashes={'result.csv':P.sha(self.repo/'result.csv')}
    def tearDown(self):self.tmp.cleanup()
    def publish(self):return P.commit_and_push(self.repo,P.BRANCH,['result.csv'],self.hashes,self.root/'transaction','Publish component\n',self.base)
    def advance_remote(self,name='remote.txt',text='unrelated remote change\n'):
        (self.other/name).write_text(text);P.git(self.other,'add','--',name);P.git(self.other,'commit','-qm','foreign work');P.git(self.other,'push','-q','origin',P.BRANCH)
        return P.git(self.other,'rev-parse','HEAD').strip()
    def assert_user_preserved(self):
        self.assertEqual((self.repo/'unrelated.txt').read_text(),'user staged\n');self.assertEqual((self.repo/'untracked.txt').read_text(),'retain user file\n');self.assertEqual((self.repo/'synrfp/morgan.py').read_text(),'old sealed source\n')
        self.assertEqual(P.git(self.repo,'diff','--cached','--name-only').splitlines(),['unrelated.txt'])
    def test_private_index_cas_preserves_unrelated_staging_work_and_core(self):
        r=self.publish();self.assertEqual(P.remote_head(self.repo,P.BRANCH),r['commit']);self.assertFalse(r['local_head_preserved']);self.assert_user_preserved()
        self.assertEqual(P.git(self.repo,'diff-tree','--no-commit-id','--name-only','-r',self.base,r['commit']).splitlines(),['result.csv'])
        self.assertIn('Publish component',P.git(self.repo,'log','-1','--format=%s'))
    def test_fresh_remote_rebase_preserves_real_head_index_and_remote_work(self):
        remote=self.advance_remote();r=self.publish();self.assertTrue(r['local_head_preserved']);self.assertEqual(P.git(self.repo,'rev-parse','HEAD').strip(),self.base);self.assertEqual(P.sha(self.index),self.index_before)
        self.assertTrue(P.ancestor(self.repo,remote,r['commit']));self.assertEqual(P.git(self.repo,'show',r['commit']+':remote.txt'),'unrelated remote change\n');self.assert_user_preserved()
    def test_foreign_same_path_edit_is_never_overwritten(self):
        remote=self.advance_remote('result.csv','mcc\n0.7\n')
        with self.assertRaisesRegex(ValueError,'Remote edited'):self.publish()
        self.assertEqual(P.remote_head(self.repo,P.BRANCH),remote);self.assertEqual(P.sha(self.index),self.index_before);self.assert_user_preserved()
    def test_resume_does_not_duplicate_commit(self):
        one=self.publish();two=self.publish();self.assertEqual(one['commit'],two['commit']);self.assertEqual(P.git(self.repo,'rev-list','--count',self.base+'..HEAD').strip(),'1');self.assert_user_preserved()
    def test_network_retry_uses_same_safe_local_commit(self):
        original=P.git;calls=[]
        def flaky(repo,*args,**kwargs):
            if args and args[0]=='push':
                calls.append(args)
                if len(calls)==1:raise RuntimeError('temporary network failure')
            return original(repo,*args,**kwargs)
        with patch.object(P,'git',side_effect=flaky):r=self.publish()
        self.assertEqual(len(calls),2);self.assertEqual(P.git(self.repo,'rev-list','--count',self.base+'..HEAD').strip(),'1');self.assert_user_preserved()
    def test_remote_race_rebases_only_approved_snapshot(self):
        original=P.git;fired=[]
        def race(repo,*args,**kwargs):
            if Path(repo)==self.repo and args and args[0]=='push'and not fired:
                fired.append(self.advance_remote())
            return original(repo,*args,**kwargs)
        with patch.object(P,'git',side_effect=race):r=self.publish()
        self.assertTrue(P.ancestor(self.repo,fired[0],r['commit']));self.assertEqual(P.git(self.repo,'show',r['commit']+':result.csv'),'mcc\n0.9\n');self.assert_user_preserved()

if __name__=='__main__':unittest.main()
