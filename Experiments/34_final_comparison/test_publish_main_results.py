"""Publication safety regressions using disposable repositories, never the real one."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('main_result_publisher',Path(__file__).with_name('publish_main_results.py'))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def receipt():
    return dict(phase='complete',configuration_selected=P.SELECTED,total_expected=P.TOTAL,
                logical_cells=P.TOTAL,total_completed=P.TOTAL,
                tasks={k:dict(complete=True,completed_cells=n,expected_cells=n) for k,n in P.EXPECTED.items()},
                selection_interpretation='exploratory_after_prior_test_results_viewed',independent_confirmation=False,
                components={k:dict(phase='complete') for k in ('regression','clustering')})


class GateTests(unittest.TestCase):
    def test_complete_exact_scientific_scope(self):
        P.check_header(receipt())

    def test_missing_clustering_cannot_publish(self):
        value = receipt()
        value['tasks']['clustering']['complete'] = False
        with self.assertRaisesRegex(ValueError,'Incomplete final task'):
            P.check_header(value)

    def test_count_only_completion_cannot_publish(self):
        value = receipt()
        value['components']['clustering']['phase'] = 'pending'
        with self.assertRaisesRegex(ValueError,'component closure'):
            P.check_header(value)

    def test_wrong_preset_or_independent_claim_rejected(self):
        for key,value in [('configuration_selected','historical1024'),('independent_confirmation',True)]:
            record = receipt()
            record[key] = value
            with self.assertRaises(ValueError):
                P.check_header(record)

    def test_allowlist_has_only_main_tables_and_metadata(self):
        paths = P.allowed_paths()
        self.assertEqual(len(paths),len(set(paths)))
        self.assertFalse(any('35_specialist_references' in x for x in paths))
        self.assertEqual(sum(p.endswith('_cells.csv') and '/clustering/' not in p for p in paths),7)
        for task in P.EXPECTED:
            self.assertTrue(any(p.endswith('/'+task+'_cells.csv') for p in paths))
            self.assertTrue(any(p.endswith('/'+task+'_summary.csv') for p in paths))
        self.assertFalse(any(Path(x).suffix not in P.ALLOWED_SUFFIXES for x in paths))

    def test_heavyweight_and_symlinked_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)
            for name in ('models/a.json','runs/a.csv','cache/a.md','a.pt','35_specialist_references/a.json'):
                f=repo/name; f.parent.mkdir(parents=True,exist_ok=True); f.write_text('data')
                with self.assertRaises(ValueError):
                    P.safe_path(repo,name)
            (repo/'real.csv').write_text('x\n1\n')
            (repo/'link.csv').symlink_to(repo/'real.csv')
            with self.assertRaises(ValueError):
                P.safe_path(repo,'link.csv')


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.repo=self.root/'repo'; self.repo.mkdir()
        subprocess.run(['git','init','-q','-b',P.BRANCH,str(self.repo)],check=True)
        P.git(self.repo,'config','user.name','Publication Test')
        P.git(self.repo,'config','user.email','test@example.invalid')
        (self.repo/'user.md').write_text('base\n')
        (self.repo/'unstaged.md').write_text('base\n')
        P.git(self.repo,'add','user.md','unstaged.md')
        P.git(self.repo,'commit','-qm','base')
        self.base=P.git(self.repo,'rev-parse','HEAD').strip()
        (self.repo/'results.csv').write_text('method,score\nsynrfp,0.7\n')
        self.hashes={'results.csv':P.sha(self.repo/'results.csv')}
        self.directory=self.root/'transaction'

    def tearDown(self):
        self.temp.cleanup()

    def commit(self):
        return P.commit_snapshot(self.repo,['results.csv'],self.hashes,self.directory,P.BRANCH)

    def test_private_index_preserves_staged_and_unstaged_user_work(self):
        (self.repo/'user.md').write_text('staged user change\n')
        P.git(self.repo,'add','user.md')
        before=P.git(self.repo,'ls-files','--stage','--','user.md')
        (self.repo/'unstaged.md').write_text('unstaged user change\n')
        result=self.commit()
        self.assertEqual(P.git(self.repo,'rev-parse','HEAD').strip(),result['commit'])
        self.assertEqual(P.git(self.repo,'ls-files','--stage','--','user.md'),before)
        self.assertEqual(P.git(self.repo,'diff','--cached','--name-only').splitlines(),['user.md'])
        self.assertEqual(P.git(self.repo,'diff','--name-only').splitlines(),['unstaged.md'])
        self.assertEqual(P.git(self.repo,'diff-tree','--no-commit-id','--name-only','-r','HEAD').splitlines(),['results.csv'])
        self.assertEqual((self.repo/'user.md').read_text(),'staged user change\n')
        self.assertEqual(P.git(self.repo,'show','HEAD:user.md'),'base\n')

    def test_concurrent_unrelated_staging_is_preserved(self):
        real=P.git
        staged=False
        def intervene(repo,*args,**kwargs):
            nonlocal staged
            value=real(repo,*args,**kwargs)
            if args[0]=='commit-tree' and not staged:
                staged=True
                (self.repo/'user.md').write_text('staged while publisher builds private tree\n')
                real(self.repo,'add','user.md')
            return value
        with patch.object(P,'git',side_effect=intervene):
            self.commit()
        self.assertEqual(P.git(self.repo,'diff','--cached','--name-only').splitlines(),['user.md'])
        self.assertEqual(P.git(self.repo,'show',':user.md'),'staged while publisher builds private tree\n')

    def test_concurrent_head_advance_refused(self):
        real=P.git
        moved=None
        def intervene(repo,*args,**kwargs):
            nonlocal moved
            value=real(repo,*args,**kwargs)
            if args[0]=='commit-tree' and moved is None:
                tree=real(repo,'rev-parse','HEAD^{tree}').strip()
                moved=real(repo,'commit-tree',tree,'-p',self.base,input=b'concurrent user commit\n').strip()
                real(repo,'update-ref','refs/heads/'+P.BRANCH,moved,self.base)
            return value
        with patch.object(P,'git',side_effect=intervene):
            with self.assertRaisesRegex(ValueError,'HEAD changed concurrently'):
                self.commit()
        self.assertEqual(P.git(self.repo,'rev-parse','HEAD').strip(),moved)
        self.assertFalse((self.directory/'COMMITTED.json').exists())
        self.assertTrue((self.repo/'results.csv').exists())

    def test_mutated_output_refused_before_commit(self):
        real=P.git
        def intervene(repo,*args,**kwargs):
            value=real(repo,*args,**kwargs)
            if args[0]=='add' and kwargs.get('index'):
                (self.repo/'results.csv').write_text('method,score\nsynrfp,0.9\n')
            return value
        with patch.object(P,'git',side_effect=intervene):
            with self.assertRaisesRegex(ValueError,'changed while staging'):
                self.commit()
        self.assertEqual(P.git(self.repo,'rev-parse','HEAD').strip(),self.base)
        self.assertFalse((self.directory/'COMMITTED.json').exists())

    def test_existing_git_lock_not_removed_or_bypassed(self):
        lock=self.repo/'.git/index.lock'; lock.write_text('other process')
        with self.assertRaises(FileExistsError):
            self.commit()
        self.assertEqual(lock.read_text(),'other process')
        self.assertEqual(P.git(self.repo,'rev-parse','HEAD').strip(),self.base)

    def test_user_staging_in_publication_path_is_not_overwritten(self):
        P.git(self.repo,'add','results.csv')
        before=P.git(self.repo,'show',':results.csv')
        (self.repo/'results.csv').write_text('method,score\nsynrfp,0.8\n')
        self.hashes['results.csv']=P.sha(self.repo/'results.csv')
        with self.assertRaisesRegex(ValueError,'User staging overlaps'):
            self.commit()
        self.assertEqual(P.git(self.repo,'show',':results.csv'),before)
        self.assertEqual(P.git(self.repo,'rev-parse','HEAD').strip(),self.base)

    def test_branch_change_refused(self):
        P.git(self.repo,'checkout','-qb','other')
        with self.assertRaisesRegex(ValueError,'Branch changed before staging'):
            self.commit()
        self.assertEqual(P.git(self.repo,'rev-parse','HEAD').strip(),self.base)

    def configure_remote(self):
        remote=self.root/'remote.git'
        subprocess.run(['git','init','-q','--bare',str(remote)],check=True)
        P.git(self.repo,'remote','add','origin',str(remote))
        P.git(self.repo,'push','-q','origin',self.base+':refs/heads/'+P.BRANCH)
        return remote

    def test_normal_push_verifies_remote_and_resumes_single_commit(self):
        remote=self.configure_remote()
        result=self.commit()
        P.write(self.directory/'GATE.json',dict(logical_cells=8238))
        with patch.multiple(P,REPO=self.repo,ROOT=self.directory,REMOTE_URL=str(remote)):
            published=P.push_committed(result,dict(publisher_sources_sha256={}))
            again=P.push_committed(result,dict(publisher_sources_sha256={}))
        self.assertEqual(published['commit'],result['commit'])
        self.assertEqual(again['commit'],result['commit'])
        self.assertEqual(P.git(remote,'rev-parse','refs/heads/'+P.BRANCH).strip(),result['commit'])
        self.assertFalse(published['weights_published'])
        self.assertFalse(published['specialist_references_waited_for'])
        self.assertEqual(P.git(self.repo,'rev-list','--count',self.base+'..HEAD').strip(),'1')

    def test_divergent_remote_is_never_force_pushed(self):
        remote=self.configure_remote()
        result=self.commit()
        tree=P.git(self.repo,'rev-parse',self.base+'^{tree}').strip()
        divergent=P.git(self.repo,'commit-tree',tree,'-p',self.base,input=b'other authorized work\n').strip()
        P.git(self.repo,'push','-q','origin',divergent+':refs/heads/'+P.BRANCH)
        P.write(self.directory/'GATE.json',dict(logical_cells=8238))
        with patch.multiple(P,REPO=self.repo,ROOT=self.directory,REMOTE_URL=str(remote)):
            with self.assertRaisesRegex(RuntimeError,'Git command failed'):
                P.push_committed(result,dict(publisher_sources_sha256={}))
        self.assertEqual(P.git(remote,'rev-parse','refs/heads/'+P.BRANCH).strip(),divergent)
        self.assertFalse((self.directory/'PUBLISHED.json').exists())
        self.assertTrue((self.directory/'COMMITTED.json').exists())

    def test_missing_completion_is_waiting_and_never_touches_git(self):
        config=dict(publisher_sources_sha256={})
        fake_run=self.root/'missing_run'; fake_export=self.root/'export'
        fake_root=self.root/'publisher'; fake_root.mkdir()
        P.write(fake_root/'config.json',config)
        with patch.multiple(P,ROOT=fake_root,RUN=fake_run,EXPORT=fake_export),\
             patch.object(P,'verify_setup'),patch.object(P,'commit_snapshot') as commit:
            result=P.once()
        self.assertEqual(result['phase'],'waiting_for_audited_7_of_7')
        commit.assert_not_called()
        self.assertEqual(P.git(self.repo,'rev-parse','HEAD').strip(),self.base)


if __name__=='__main__':
    unittest.main()
