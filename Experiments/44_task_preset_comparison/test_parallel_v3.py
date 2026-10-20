"""Independent-fit partition, provenance, handoff and resource controls."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=load('five44_controller_tests_target',HERE/'controller_parallel_v3.py');P=T.P

def fixture():
    tasks=[];weights={}
    for i in range(12):
        for seed in range(10):
            for pop in ('all_rows','shared_valid_all_six'):
                origin='population_identical_copy'if i<4 and pop!='all_rows'else'fresh'
                t=dict(dataset='d'+str(i),population=pop,seed=seed,origin=origin);tasks.append(t)
                weights[P.key(t)]=.1 if origin!='fresh'else 10*(i+1)+seed
    a,loads=P.lpt(tasks,weights)
    plan=dict(schema='task-preset44-five-shard-plan-v3',workers=5,threads=1,memory_gib=7,assignments=a)
    return dict(tasks=tasks),weights,plan,loads

class FiveShardControls(unittest.TestCase):
    def test_exact_240_union_200_fits_and_40_copies(self):
        scientific,weights,plan,_=fixture();self.assertIs(P.validate_partition(plan,scientific),plan)
        self.assertEqual(sum(len(x)for x in plan['assignments'].values()),240)
        self.assertEqual(sum(t['origin']=='fresh'for x in plan['assignments'].values()for t in x),200)
    def test_lpt_deterministic_and_balanced_without_splitting_pairs(self):
        scientific,weights,plan,loads=fixture();a,b=P.lpt(list(reversed(scientific['tasks'])),weights)
        self.assertEqual(a,plan['assignments']);self.assertEqual(b,loads)
        self.assertLessEqual(max(loads.values())-min(loads.values()),max(weights.values())*2)
    def test_pair_dependency_foreign_tasks_or_duplicates_refused(self):
        scientific,weights,plan,_=fixture()
        for mode in ('split','foreign','duplicate'):
            p=copy.deepcopy(plan);first=p['assignments']['0']
            if mode=='split':first[0],first[1]=first[1],first[0]
            elif mode=='foreign':first[0]['dataset']='foreign'
            else:first[2]=copy.deepcopy(first[0])
            with self.assertRaises(ValueError):P.validate_partition(p,scientific)
    def test_all_population_pairs_remain_on_one_worker(self):
        _,_,p,_=fixture();owners={}
        for worker,tasks in p['assignments'].items():
            for t in tasks:
                k=t['dataset'],t['seed'];self.assertEqual(owners.setdefault(k,worker),worker)
    def test_wrong_worker_count_or_memory_profile_refused(self):
        scientific,_,p,_=fixture()
        for field,value in (('workers',4),('threads',2),('memory_gib',8)):
            changed=copy.deepcopy(p);changed[field]=value
            with self.assertRaises(ValueError):P.validate_partition(changed,scientific)
    def test_foreign_or_undefined_timing_weight_refused(self):
        scientific,w,_,_=fixture();w[P.key(scientific['tasks'][0])]=float('nan')
        with self.assertRaises(ValueError):P.lpt(scientific['tasks'],w)
    def test_five_distinct_single_cpu_seven_gib_jobs(self):
        execution=dict(partition_plan_sha256='a'*64,sources_sha256={})
        jobs=[T.job_spec(i,1,execution)for i in range(5)]
        self.assertEqual(len({j['cores']for j in jobs}),5)
        self.assertEqual(sum(j['cpu_equivalents']for j in jobs),5)
        self.assertEqual(sum(j['memory_gib']for j in jobs),35)
        self.assertTrue(all(j['memory_gib']==7 and j['environment']['CUDA_VISIBLE_DEVICES']==''and'gpu'not in j for j in jobs))
    def test_new_generations_preserve_invocation_paths(self):
        e=dict(partition_plan_sha256='a'*64,sources_sha256={})
        self.assertNotEqual(T.job_spec(0,1,e)['run_id'],T.job_spec(0,2,e)['run_id'])
    def test_aggregate_cpu_and_memory_caps_still_enforced(self):
        g=T.O.GIB;v=dict(physical_cores=32,memory_total_bytes=125*g,memory_available_bytes=100*g,
                       cpu_cap_sum=22.5,memory_cap_sum_bytes=93.25*g,disk_free_bytes=50*g)
        self.assertEqual(T.O.R.admission(v,1,7*g,dict(active=False)),[])
        v['memory_cap_sum_bytes']=101*g
        self.assertIn('aggregate_memory_85pct',T.O.R.admission(v,1,7*g,dict(active=False)))
    def test_missing_root_review_never_dispatches(self):
        with tempfile.TemporaryDirectory()as temp,patch.object(T,'ROOT',Path(temp)),patch.object(T.C,'ensure')as ensure:
            self.assertEqual(T.cycle()['phase'],'waiting_tested_five_shard_review');ensure.assert_not_called()
    def test_old_live_cohort_never_allows_new_scientific_dispatch(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);T.O.write(root/'clustering/plan.json',{});T.O.write(root/'control/clustering_execution.json',{})
            with patch.object(T,'ROOT',root),patch.object(T.O,'dispatched',return_value=[dict(stage='clustering',unit='old')]), \
                 patch.object(T.C,'props',return_value=dict(ActiveState='active',MainPID='9')):
                self.assertFalse(T.old_workers_done())
    def test_unattested_or_foreign_jobs_cannot_receive_yield(self):
        with patch.object(T.subprocess,'run')as run:
            with self.assertRaises(T.C.IntegrityError):T.request_yield(dict(id='morgan43',run_id='v3-w0-g0001'))
            run.assert_not_called()
    def test_worker_receipts_do_not_change_scientific_math_or_thread_history(self):
        text=(HERE/'worker_parallel_v3.py').read_text()
        self.assertIn('C.run(root,worker=worker,workers=5,threads=1,run_id=run_id)',text)
        self.assertNotIn('C.fit=',text);self.assertNotIn('C.verify_result=',text)
        self.assertIn("'OPERATIONAL.json'",text)
    def test_governor_uses_original_lock_and_keeps_old_controllers_sealed(self):
        text=(HERE/'controller_parallel_v3.py').read_text()
        self.assertIn("ROOT/'control/controller.lock'",text)
        self.assertNotIn("['systemctl','--user','stop'",text)
        self.assertNotIn('lease_available(',text)
    def test_report_closure_waits_for_actual_report_process_exit(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);P.write(root/'parallel_v3/REPORT_COMPLETE.json',dict(phase='complete'))
            with patch.object(T,'ROOT',root),patch.object(T.O,'tree_audit'), \
                 patch.object(T.C,'props',return_value=dict(ActiveState='inactive',MainPID='998')):
                self.assertEqual(T.report(dict(partition_plan_sha256='a'*64))['phase'],'waiting_report_service_exit')
    def test_resource_pressure_retains_previous_hysteresis(self):
        text=(HERE/'controller_parallel_v3.py').read_text()
        self.assertIn("O.read(previous).get('held',False)",text)
        self.assertIn('O.write(previous,dict(pressure',text)

if __name__=='__main__':unittest.main()
