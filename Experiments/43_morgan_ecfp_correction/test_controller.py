"""Operational mutation and capacity controls; no fits, GPU probes or launches."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('morgan43_controller_tests_target',HERE/'controller.py')
T=importlib.util.module_from_spec(s);s.loader.exec_module(T)

class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.control=self.root/'control'
        self.stack=[]
        for obj,name,value in ((T,'ROOT',self.root),(T,'CONTROL',self.control),(T,'SPECIALISTS',self.root/'35'),(T,'D',T.C.Digests())):
            p=patch.object(obj,name,value);p.start();self.addCleanup(p.stop)
        self.execution=dict(plan_sha256='a'*64,sources_sha256={str(T.ENTRY):T.B.sha(T.ENTRY)})
    def put(self,path,v):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v));return path
    def plan(self):
        tasks=[dict(task='clustering',dataset='d',config='x',seed=i) for i in (0,1)]
        return dict(sources_sha256=self.execution['sources_sha256'],sources={'a':dict(feature_receipt=str(self.root/'features/a/COMPLETE.json'))},panels={'d':dict(n=2)},tasks=dict(clustering=tasks,classification=[dict(task='classification',dataset='d',config='x',fold=0)]))
    def terminal(self,job,plan,kind='complete'):
        paths=T.expected_paths(plan,job['lane']);catalog={}
        for p in paths:self.put(p,dict(phase='complete'));catalog[p]=T.B.sha(p)
        return dict(phase=kind,stage=job['stage'],worker=job['worker'],workers=job['workers'],run_id=job['run_id'],plan_sha256=self.execution['plan_sha256'],sources_sha256=self.execution['sources_sha256'],expected_items=len(paths),audited_items=len(paths),all_shard_items_verified=True,outputs_sha256=catalog)
    def value(self):
        return dict(physical_cores=32,memory_total_bytes=125*T.GIB,memory_available_bytes=80*T.GIB,cpu_cap_sum=4,memory_cap_sum_bytes=16*T.GIB,disk_free_bytes=50*T.GIB,units=[])
    def test_exact_profiles_and_gpu0_absent(self):
        for lane,(stage,index,workers,cpu,mem,cores) in T.LANES.items():
            j=T.worker_spec(lane,1,self.execution)
            self.assertEqual((j['cpu_equivalents'],j['memory_gib'],j['cores']),(cpu,mem,cores))
            self.assertEqual(j.get('gpu'),1 if stage=='mlp' else None)
            self.assertEqual(j['environment']['CUDA_VISIBLE_DEVICES'],'1' if stage=='mlp' else '')
        self.assertIn('--encode-processes',T.worker_spec('e0',1,self.execution)['arguments'])
    def test_fresh_generation_paths(self):
        a=T.worker_spec('c0',1,self.execution);b=T.worker_spec('c0',2,self.execution)
        self.assertNotEqual(a['run_id'],b['run_id']);self.assertNotEqual(a['unit'],b['unit'])
    def test_nested_model_artifact_mutation(self):
        p=self.root/'models/x/result.json';attempt=p.parent/'attempt-001';weight=attempt/'model.bin';weight.parent.mkdir(parents=True);weight.write_bytes(b'original')
        self.put(p,dict(phase='complete',attempt_directory=str(attempt),artifacts_sha256={'model.bin':T.B.sha(weight)}))
        root=self.put(self.root/'COMPLETE.json',dict(outputs_sha256={str(p):T.B.sha(p)}));T.tree_audit(root)
        weight.write_bytes(b'changed')
        with self.assertRaises(T.C.IntegrityError):T.tree_audit(root)
    def test_nested_feature_catalog_mutation(self):
        f=self.root/'features.npz';f.write_bytes(b'x');r=self.put(self.root/'features.json',dict(files={'features.npz':T.B.sha(f)}));T.tree_audit(r);f.write_bytes(b'y')
        with self.assertRaises(T.C.IntegrityError):T.tree_audit(r)
    def test_cluster_assignment_and_centers_hashes(self):
        r=self.root/'seed.json';a=r.with_suffix('.npz');c=r.with_suffix('.model.npz');a.write_bytes(b'a');c.write_bytes(b'c')
        self.put(r,dict(assignments_sha256=T.B.sha(a),centers_sha256=T.B.sha(c)));T.tree_audit(r);c.write_bytes(b'd')
        with self.assertRaises(T.C.IntegrityError):T.tree_audit(r)
    def test_full_shard_requires_exact_key_set(self):
        plan=self.plan();j=T.worker_spec('c0',1,self.execution);r=self.terminal(j,plan);p=self.put(self.root/'term.json',r)
        self.assertEqual(T.validate_terminal(p,j,self.execution,plan),'complete')
        r['outputs_sha256']={};r['audited_items']=0;self.put(p,r)
        with self.assertRaises(T.C.IntegrityError):T.validate_terminal(p,j,self.execution,plan)
    def test_terminal_wrong_identity_refused(self):
        plan=self.plan();j=T.worker_spec('m0',1,self.execution);r=self.terminal(j,plan);r['plan_sha256']='b'*64;p=self.put(self.root/'term.json',r)
        with self.assertRaises(T.C.IntegrityError):T.validate_terminal(p,j,self.execution,plan)
    def test_yielded_partial_is_not_completion(self):
        plan=self.plan();j=T.worker_spec('m0',1,self.execution);r=self.terminal(j,plan,'yielded');r.update(outputs_sha256={},audited_items=0,all_shard_items_verified=False);p=self.put(self.root/'term.json',r)
        self.assertEqual(T.validate_terminal(p,j,self.execution,plan),'yielded')
        with self.assertRaises(T.C.IntegrityError):T.validate_terminal(p,j,self.execution,plan,complete=True)
    def test_existing_invocation_preserved_with_new_generation(self):
        j=T.worker_spec('c0',1,self.execution);(self.root/'workers'/j['run_id']).mkdir(parents=True)
        p=dict(ActiveState='inactive',InvocationID='old',Result='success')
        with patch.object(T.C,'props',return_value=p),patch.object(T.C,'failure_logs',return_value=''),patch.object(T,'terminal',return_value=None),patch.object(T.time,'time',return_value=100):
            current,gen,blocked=T.latest_job('c0',[j],self.execution,self.plan());self.assertIsNone(current);self.assertEqual(gen,1);self.assertEqual(blocked['phase'],'retry_backoff')
        with patch.object(T.C,'props',return_value=p),patch.object(T,'terminal',return_value=None),patch.object(T.time,'time',return_value=131):
            current,gen,blocked=T.latest_job('c0',[j],self.execution,self.plan());self.assertIsNone(current);self.assertIsNone(blocked)
        self.assertTrue((self.root/'workers'/j['run_id']).exists());self.assertTrue((self.control/'abandoned_runs'/(j['run_id']+'.json')).exists())
    def test_oom_failure_latches_lane(self):
        j=T.worker_spec('e0',1,self.execution);p=dict(ActiveState='failed',InvocationID='old',Result='oom-kill')
        with patch.object(T.C,'props',return_value=p),patch.object(T.C,'failure_logs',return_value=''),patch.object(T,'terminal',return_value=None):
            _,_,b=T.latest_job('e0',[j],self.execution,self.plan());self.assertEqual(b,dict(phase='latched',reason='cgroup_oom'))
    def test_no_started_invocation_admission_reuses_only_unlaunched_id(self):
        j=T.worker_spec('e0',1,self.execution)
        with patch.object(T.C,'props',return_value=dict(ActiveState='inactive',InvocationID='')),patch.object(T,'terminal',return_value=None):
            self.assertEqual(T.latest_job('e0',[j],self.execution,self.plan())[0],j)
    def test_exact_reservation_visibility_and_suffix(self):
        j=T.worker_spec('c0',1,self.execution);v=self.value();v['units']=[dict(unit=j['unit']+'.service',cpu_equivalents=4,memory_max_bytes=12*T.GIB)]
        with patch.object(T,'snapshot',return_value=v),patch.object(T.C,'props',return_value={}),patch.object(T.C,'attest_service'):
            self.assertTrue(T.reservation_visible(j,self.execution,self.plan()))
        v['units'][0]['memory_max_bytes']=11*T.GIB
        with patch.object(T,'snapshot',return_value=v):
            with self.assertRaises(T.C.IntegrityError):T.reservation_visible(j,self.execution,self.plan())
    def test_missing_reservation_blocks_next_launch(self):
        j=T.worker_spec('c0',1,self.execution)
        with patch.object(T,'snapshot',return_value=self.value()),patch.object(T.C,'props',return_value=dict(ActiveState='active')),patch.object(T.time,'sleep'):
            with self.assertRaises(T.C.IntegrityError):T.reservation_visible(j,self.execution,self.plan())
    def test_85pct_aggregate_and_old41_headroom(self):
        v=self.value();v['cpu_cap_sum']=27;v['memory_cap_sum_bytes']=106*T.GIB
        self.assertIn('aggregate_cpu_85pct',T.R.admission(v,1,0))
        self.assertIn('aggregate_memory_85pct',T.R.admission(v,0,1*T.GIB))
        v=self.value();v['memory_cap_sum_bytes']=50*T.GIB
        self.assertIn('clustering_priority_memory_headroom',T.R.admission(v,4,12*T.GIB,dict(active=True,pending_cpu=2,pending_memory_bytes=32*T.GIB)))
    def test_disk_recovery_50_43_44point5_gib(self):
        v=self.value();self.assertFalse(T.pressure_state(v,True)['yield_required'])
        v['disk_free_bytes']=41*T.GIB;self.assertTrue(T.pressure_state(v,True)['yield_required'])
        v['disk_free_bytes']=42.5*T.GIB;self.assertTrue(T.pressure_state(v,True,True)['yield_required'])
        v['disk_free_bytes']=44.5*T.GIB;self.assertFalse(T.pressure_state(v,True,True)['yield_required'])
        v['disk_free_bytes']=39*T.GIB;self.assertIn('disk_reserve_40GiB',T.R.admission(v,0,0))
    def test_ram_30_35_hysteresis(self):
        v=self.value();v['memory_available_bytes']=.29*v['memory_total_bytes'];self.assertTrue(T.pressure_state(v,True)['yield_required'])
        v['memory_available_bytes']=.34*v['memory_total_bytes'];self.assertTrue(T.pressure_state(v,True,True)['yield_required'])
        v['memory_available_bytes']=.36*v['memory_total_bytes'];self.assertFalse(T.pressure_state(v,True,True)['yield_required'])
    def t5(self,phase='complete'):
        base=T.SPECIALISTS;native=base/'t5chem/models/native_seed43/COMPLETE.json';artifact=native.parent/'best.pt';artifact.parent.mkdir(parents=True);artifact.write_bytes(b'checkpoint')
        self.put(native,dict(phase='complete',split='native_seed43',development_replay=True,artifacts_sha256={str(artifact):T.B.sha(artifact)}))
        plan=self.put(base/'t5chem/plan.json',dict(data='frozen'));ticket=self.put(base/'control/tickets/t5chem/002.json',dict(phase=phase,plan_sha256=T.B.sha(plan),completed_model_ids=['native_seed43'],artifacts_sha256={str(native):T.B.sha(native)}))
        self.put(base/'control/launches'/(T.T5_UNIT+'.json'),dict(spec=dict(unit=T.T5_UNIT,gpu=1,sources_sha256=self.execution['sources_sha256'])))
        return artifact,ticket
    def test_t5_raw_ticket_native_nested_and_lease_required(self):
        artifact,ticket=self.t5()
        with patch.object(T.C,'props',return_value=dict(ActiveState='inactive',MainPID='0')),patch.object(T.C,'attest_service'),patch.object(T.B,'lease_available',return_value=True):
            self.assertTrue(T.t5_finished());artifact.write_bytes(b'mutated')
            with self.assertRaises(T.C.IntegrityError):T.t5_finished()
    def test_t5_running_or_lease_busy_never_admits(self):
        _,ticket=self.t5('running')
        with patch.object(T.C,'props',return_value=dict(ActiveState='inactive',MainPID='0')),patch.object(T.C,'attest_service'),patch.object(T.B,'lease_available',return_value=True):self.assertFalse(T.t5_finished())
        r=T.read(ticket);r['phase']='complete';self.put(ticket,r)
        with patch.object(T.C,'props',return_value=dict(ActiveState='inactive',MainPID='0')),patch.object(T.C,'attest_service'),patch.object(T.B,'lease_available',return_value=False):self.assertFalse(T.t5_finished())
        with patch.object(T.C,'props',return_value=dict(ActiveState='active',MainPID='12')),patch.object(T.C,'attest_service'):self.assertFalse(T.t5_finished())
    def test_stale_started_pid_cannot_authorize_signal(self):
        j=T.worker_spec('c0',1,self.execution);self.put(self.root/'workers'/j['run_id']/'STARTED.json',dict(phase='ready',run_id=j['run_id'],stage='clustering',pid=10))
        with patch.object(T.C,'props',return_value=dict(ActiveState='active',MainPID='11')),patch.object(T.C,'attest_service'),patch.object(T.subprocess,'run') as run:
            self.assertFalse(T.request_yield(j));run.assert_not_called()
    def test_signal_other_job_refused(self):
        with self.assertRaises(T.C.IntegrityError):T.request_yield(dict(lane='foreign',id='other'))
    def test_immutable_dispatch_seal_rejects_changed_identity(self):
        j=T.worker_spec('c0',1,self.execution);T.publish(j);j['memory_gib']=999
        with self.assertRaises(T.C.IntegrityError):T.publish(j)

    def test_nested_history_list_is_an_artifact_not_receipt(self):
        h=self.put(self.root/'history.json',[dict(epoch=1,loss=1.5)])
        r=self.put(self.root/'COMPLETE.json',dict(artifacts_sha256={str(h):T.B.sha(h)}));T.tree_audit(r)
    def test_storage_symlink_retarget_rejected(self):
        a=self.root/'a';b=self.root/'b';a.mkdir();b.mkdir();(self.root/'artifacts').symlink_to(a)
        self.put(self.root/'ARTIFACT_STORAGE.json',dict(phase='complete',existing_files_moved_or_deleted=False,logical_artifacts_path=str(self.root/'artifacts'),physical_artifacts_path=str(a)))
        T.storage_guard();(self.root/'artifacts').unlink();(self.root/'artifacts').symlink_to(b)
        with self.assertRaises(T.C.IntegrityError):T.storage_guard()
    def test_report_requires_two_exact_task_closures_and_nested_files(self):
        outputs={}
        for task in ('classification','clustering'):
            artifact=self.root/'report'/task/'table.csv';artifact.parent.mkdir(parents=True);artifact.write_text('full')
            p=self.put(artifact.parent/'COMPLETE.json',dict(phase='complete',outputs_sha256={'table.csv':T.B.sha(artifact)}));outputs[str(p)]=T.B.sha(p)
        v=dict(schema='morgan43-full-correction-complete-v1',phase='complete',corrected_mlp_cells=1680,corrected_clustering_cells=1440,combined_classification_cells=5040,combined_clustering_cells=4320,all_keyed_closures_verified=True,outputs_sha256=outputs,plan_sha256=self.execution['plan_sha256'],tokenizer_profile='synrfp-ecfp-count-v1')
        self.put(self.root/'COMPLETE.json',v);self.assertTrue(T.report_complete(self.execution));artifact.write_text('changed')
        with self.assertRaises(T.C.IntegrityError):T.report_complete(self.execution)
    def test_report_retry_allocates_new_generation(self):
        j=dict(id='morgan43-r0-g0001',unit='synrfp-morgan43-r0-g0001-x',sources_sha256=self.execution['sources_sha256'],lane='r0',stage='report',run_id='r0-g0001',generation=1)
        p=self.put(self.control/'report_dispatch/r0-g0001.json',j);self.put(self.control/'report_dispatch_seals'/p.name,dict(sha256=T.B.sha(p)))
        self.put(self.control/'abandoned_runs/r0-g0001.json',dict(reason=None,retry_after_epoch=0))
        with patch.object(T.C,'props',return_value=dict(ActiveState='inactive',InvocationID='old')),patch.object(T.C,'attest_service'),patch.object(T.C,'ensure',return_value=dict(phase='waiting_resources')) as ensure:
            T.ensure_report(self.execution,self.plan());self.assertEqual(ensure.call_args.args[0]['run_id'],'r0-g0002')
    def test_reference_cannot_resume_before_mlp_complete(self):
        self.put(self.root/'operations/REFERENCE_RESUME_READY.json',dict(unit=T.PAUSED_GOVERNORS[1]))
        with patch.object(T.C,'props',return_value=dict(ActiveState='inactive')),patch.object(T.subprocess,'run') as run:
            self.assertEqual(T.resume_references(self.execution,False,[])['phase'],'waiting_all_1680_mlp');run.assert_not_called()
    def test_reference_waits_for_all_mlp_pids_to_exit(self):
        self.put(self.root/'operations/REFERENCE_RESUME_READY.json',dict(unit=T.PAUSED_GOVERNORS[1]))
        with patch.object(T.C,'props',return_value=dict(ActiveState='inactive',MainPID='999')),patch.object(T.subprocess,'run') as run:
            self.assertEqual(T.resume_references(self.execution,True,[dict(stage='mlp',unit='owned')])['phase'],'waiting_mlp_unit_exit');run.assert_not_called()
    def test_reference_governor_exact_argv_caps_and_unit_hash(self):
        unit=self.root/'governor.service';unit.write_text('ExecStart=/usr/bin/taskset -c 30 /python -u /entry.py\n')
        ready=dict(unit=T.PAUSED_GOVERNORS[1],unit_file=str(unit),installed_unit_sha256=T.B.sha(unit),sources_sha256={str(unit):T.B.sha(unit)})
        props=dict(LoadState='loaded',ExecStart='{ argv[]=/usr/bin/taskset -c 30 /python -u /entry.py ; ignore_errors=no ; }',CPUQuotaPerSecUSec='100ms',MemoryMax=str(256*1024**2),KillMode='control-group',WorkingDirectory=str(T.REPO),Environment='CUDA_VISIBLE_DEVICES=')
        T.attest_reference_governor(ready,props);props['MemoryMax']='100'
        with self.assertRaises(T.C.IntegrityError):T.attest_reference_governor(ready,props)
    def test_resume_command_only_exact_attested_governor(self):
        self.put(self.root/'operations/REFERENCE_RESUME_READY.json',dict(unit=T.PAUSED_GOVERNORS[1]));self.put(self.control/'execution.json',self.execution);self.put(self.root/'mlp_worker_0_COMPLETE.json',dict(phase='complete'))
        inactive=dict(ActiveState='inactive',MainPID='0');active=dict(ActiveState='active',MainPID='20')
        v=self.value();v['units']=[dict(unit=T.PAUSED_GOVERNORS[1],cpu_equivalents=.1,memory_max_bytes=256*1024**2)]
        with patch.object(T.C,'props',side_effect=[inactive,active]),patch.object(T,'attest_reference_governor'),patch.object(T.B,'lease_available',return_value=True),patch.object(T,'admission',return_value=[]),patch.object(T,'snapshot',return_value=v),patch.object(T.subprocess,'run') as run:
            self.assertEqual(T.resume_references(self.execution,True,[])['phase'],'complete');run.assert_called_once_with(['systemctl','--user','enable','--now',T.PAUSED_GOVERNORS[1]],check=True)
        self.assertFalse(any('42' in x for x in run.call_args.args[0]))

    def test_report_wrong_plan_and_profile_refused(self):
        for name,value in (('plan_sha256','other'),('tokenizer_profile','old')):
            r=dict(schema='morgan43-full-correction-complete-v1',phase='complete',corrected_mlp_cells=1680,corrected_clustering_cells=1440,combined_classification_cells=5040,combined_clustering_cells=4320,all_keyed_closures_verified=True,outputs_sha256={},plan_sha256=self.execution['plan_sha256'],tokenizer_profile='synrfp-ecfp-count-v1');r[name]=value
            self.put(self.root/'COMPLETE.json',r)
            with self.assertRaises(T.C.IntegrityError):T.report_complete(self.execution)
    def test_completed_report_keeps_polling_reference_restore(self):
        self.put(self.root/'plan.json',self.plan());self.put(self.root/'mlp_worker_0_COMPLETE.json',dict(phase='complete'))
        for stage,index,*_ in T.LANES.values():self.put(self.root/(stage+'_worker_'+str(index)+'_COMPLETE.json'),dict(phase='complete'))
        with patch.object(T,'manifest',return_value=self.execution),patch.object(T,'pins'),patch.object(T.C,'props',return_value=dict(ActiveState='inactive')),patch.object(T,'dispatched',return_value=[]),patch.object(T,'lane_complete',return_value=True),patch.object(T,'resume_references',return_value=dict(phase='waiting_resources')),patch.object(T,'ensure_report',return_value=dict(phase='complete')),patch.object(T,'report_complete',return_value=True):
            self.assertEqual(T.cycle()['phase'],'report_complete_waiting_reference_restore')
    def test_stage_interpreters_match_original_cpu_and_cuda_roles(self):
        for lane,(stage,*_) in T.LANES.items():
            expected='synprop' if stage=='mlp' else 'synrfp';self.assertIn('/envs/'+expected+'/',T.worker_spec(lane,1,self.execution)['python'])

    def test_waiting_mlp_yields_only_needed_owned_cluster(self):
        jobs=[T.worker_spec('c0',1,self.execution),T.worker_spec('c1',1,self.execution)]
        v=self.value();v['memory_cap_sum_bytes']=70*T.GIB
        def admit(value,cpu,mem):return ['memory'] if value['memory_cap_sum_bytes']+mem>78*T.GIB else []
        with patch.object(T.C,'props',return_value=dict(ActiveState='active')),patch.object(T,'admission',side_effect=admit),patch.object(T,'request_yield') as yielding:
            self.assertEqual(T.yield_for_mlp(jobs,v),['c1-g0001']);yielding.assert_called_once_with(jobs[1])
    def test_no_pointless_yield_if_mlp_still_cannot_fit(self):
        jobs=[T.worker_spec('c1',1,self.execution)]
        with patch.object(T.C,'props',return_value=dict(ActiveState='active')),patch.object(T,'admission',return_value=['memory']),patch.object(T,'request_yield') as yielding:
            self.assertEqual(T.yield_for_mlp(jobs,self.value()),[]);yielding.assert_not_called()

if __name__=='__main__':unittest.main()
