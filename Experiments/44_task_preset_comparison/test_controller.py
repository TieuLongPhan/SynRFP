"""Operational mutation controls; never install a unit or execute a real fit."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('preset44_controller_test_target', HERE / 'controller.py')
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.control = self.root / 'control'
        for obj, name, value in ((T, 'ROOT', self.root), (T, 'CONTROL', self.control),
                                 (T, 'ROOT43', self.root / '43'), (T.R, 'PRIORITY_ROOT', self.root / '41'),
                                 (T, 'D', T.C.Digests()), (T, 'DECISION_CACHE', T.U.HashCache())):
            current = patch.object(obj, name, value)
            current.start()
            self.addCleanup(current.stop)
        self.science = {str(T.ENTRY): T.B.sha(T.ENTRY)}
        self.execution = dict(plan_sha256='a' * 64, decision_sha256='b' * 64, sources_sha256=self.science)
        paused = patch.object(T.P43, 'paused41', return_value=False)
        paused.start()
        self.addCleanup(paused.stop)

    def put(self, path, value):
        T.write(path, value)
        return Path(path)

    def value(self):
        return dict(physical_cores=32, memory_total_bytes=125 * T.GIB, memory_available_bytes=100 * T.GIB,
                    cpu_cap_sum=4, memory_cap_sum_bytes=16 * T.GIB, disk_free_bytes=50 * T.GIB, units=[])

    def plan(self):
        return dict(tasks=[dict(dataset='d', population='all_rows', seed=0)], sources_sha256=self.science)

    def terminal_value(self, kind='complete'):
        plan, job = self.plan(), T.worker_spec(1, self.execution)
        outputs = {}
        for path in T.expected_paths(plan):
            self.put(path, dict(phase='complete'))
            outputs[path] = T.B.sha(path)
        return job, plan, dict(phase=kind, stage='clustering', run_id=job['run_id'], worker=0, workers=1,
                    plan_sha256=self.execution['plan_sha256'], decision_sha256=self.execution['decision_sha256'],
                    sources_sha256=self.science, expected_items=len(outputs), audited_items=len(outputs),
                    all_shard_items_verified=kind == 'complete', outputs_sha256=outputs)

    def test_one_cpu_lane_and_no_gpu_dispatch(self):
        job = T.worker_spec(1, self.execution)
        self.assertEqual((job['cpu_equivalents'], job['memory_gib'], job['cores']), (1, 12, '18'))
        self.assertEqual((job['worker'], job['workers']), (0, 1))
        self.assertNotIn('gpu', job)
        self.assertEqual(job['environment']['CUDA_VISIBLE_DEVICES'], '')
        self.assertIn('--threads', job['arguments'])
        self.assertEqual(job['arguments'][job['arguments'].index('--threads') + 1], '1')

    def test_preparation_is_capped_service_not_controller_import(self):
        self.put(self.control / 'execution.json', self.execution)
        job = T.auxiliary_spec('prepare', 1, self.execution, self.root / 'decision.json')
        self.assertEqual((job['cpu_equivalents'], job['memory_gib']), (1, 12))
        self.assertIn('--prepare', job['arguments'])
        self.assertNotIn('gpu', job)

    def test_no_classification_or_gpu_auxiliary_stage(self):
        for stage in ('mlp', 'knn', 'encode', 'gpu'):
            with self.assertRaises(T.C.IntegrityError):
                T.auxiliary_spec(stage, 1, self.execution)

    def test_generations_are_distinct(self):
        first, second = T.worker_spec(1, self.execution), T.worker_spec(2, self.execution)
        self.assertNotEqual(first['unit'], second['unit'])
        self.assertNotEqual(first['run_id'], second['run_id'])

    def test_immutable_dispatch_cannot_change_cpu(self):
        job = T.worker_spec(1, self.execution)
        T.publish(job)
        job['cpu_equivalents'] = 10
        with self.assertRaises(T.C.IntegrityError):
            T.publish(job)

    def test_no_review_means_no_dispatch_or_decision(self):
        with patch.object(T, 'decision') as decision, patch.object(T.C, 'ensure') as ensure:
            self.assertEqual(T.cycle()['phase'], 'waiting_source_bound_root_review')
            decision.assert_not_called()
            ensure.assert_not_called()

    def test_full_shard_must_have_exact_key_population(self):
        job, plan, value = self.terminal_value()
        path = self.put(self.root / 'COMPLETE.json', value)
        self.assertEqual(T.validate_terminal(path, job, self.execution, plan), 'complete')
        value.update(outputs_sha256={}, audited_items=0)
        self.put(path, value)
        with self.assertRaises(T.C.IntegrityError):
            T.validate_terminal(path, job, self.execution, plan)

    def test_shard_wrong_plan_or_seed_population_refused(self):
        for field in ('plan_sha256', 'decision_sha256', 'workers', 'run_id'):
            job, plan, value = self.terminal_value()
            value[field] = 2 if field == 'workers' else 'foreign'
            path = self.put(self.root / 'terminal.json', value)
            with self.assertRaises(T.C.IntegrityError):
                T.validate_terminal(path, job, self.execution, plan)

    def test_partial_yield_is_not_complete(self):
        job, plan, value = self.terminal_value('yielded')
        value.update(outputs_sha256={}, audited_items=0)
        path = self.put(self.root / 'YIELDED.json', value)
        self.assertEqual(T.validate_terminal(path, job, self.execution, plan), 'yielded')
        value['phase'] = 'complete'
        self.put(path, value)
        with self.assertRaises(T.C.IntegrityError):
            T.validate_terminal(path, job, self.execution, plan)

    def test_saved_nested_artifact_mutation_is_detected(self):
        weight = self.root / 'assignment.npz'
        weight.write_bytes(b'original')
        result = self.put(self.root / 'result.json', dict(artifacts_sha256={str(weight): T.B.sha(weight)}))
        T.tree_audit(result)
        weight.write_bytes(b'mutation')
        with self.assertRaises(T.C.IntegrityError):
            T.tree_audit(result)

    def test_nested_list_json_is_data(self):
        history = self.put(self.root / 'history.json', [dict(epoch=1)])
        result = self.put(self.root / 'result.json', dict(artifacts_sha256={str(history): T.B.sha(history)}))
        T.tree_audit(result)

    def test_own_only_signalling_refuses_morgan_and_reference(self):
        for job in (dict(id='morgan43-c0', run_id='c0', stage='clustering'),
                    dict(id='preset44-m0', run_id='m0', stage='mlp')):
            with patch.object(T.subprocess, 'run') as run:
                with self.assertRaises(T.C.IntegrityError):
                    T.request_yield(job)
                run.assert_not_called()

    def test_stale_pid_never_signalled(self):
        job = T.worker_spec(1, self.execution)
        self.put(self.root / 'workers' / job['run_id'] / 'STARTED.json', dict(pid=10))
        with patch.object(T.C, 'props', return_value=dict(ActiveState='active', MainPID='11')), \
             patch.object(T.C, 'attest_service'), patch.object(T.subprocess, 'run') as run:
            self.assertFalse(T.request_yield(job))
            run.assert_not_called()

    def test_original_invocation_retained_and_new_generation_after_failure(self):
        job = T.worker_spec(1, self.execution)
        (self.root / 'workers' / job['run_id']).mkdir(parents=True)
        with patch.object(T.C, 'props', return_value=dict(ActiveState='inactive', InvocationID='original', Result='exit-code')), \
             patch.object(T.C, 'failure_logs', return_value=''), patch.object(T, 'terminal', return_value=None), \
             patch.object(T.time, 'time', return_value=100):
            current, generation, state = T.latest_job('clustering', [job], self.execution, self.plan())
            self.assertIsNone(current)
            self.assertEqual(generation, 1)
            self.assertEqual(state['phase'], 'retry_backoff')
        self.assertTrue((self.root / 'workers' / job['run_id']).exists())
        with patch.object(T.C, 'props', return_value=dict(ActiveState='inactive', InvocationID='original')), \
             patch.object(T, 'terminal', return_value=None), patch.object(T.time, 'time', return_value=131):
            self.assertEqual(T.latest_job('clustering', [job], self.execution, self.plan()), (None, 1, None))

    def test_oom_failure_is_latched(self):
        job = T.worker_spec(1, self.execution)
        with patch.object(T.C, 'props', return_value=dict(ActiveState='failed', InvocationID='oom', Result='oom-kill')), \
             patch.object(T.C, 'failure_logs', return_value=''), patch.object(T, 'terminal', return_value=None):
            self.assertEqual(T.latest_job('clustering', [job], self.execution, self.plan())[2]['reason'], 'cgroup_oom')

    def test_unstarted_invocation_can_be_admitted_without_retry_id_change(self):
        job = T.worker_spec(1, self.execution)
        with patch.object(T.C, 'props', return_value=dict(ActiveState='inactive', InvocationID='')), \
             patch.object(T, 'terminal', return_value=None):
            self.assertEqual(T.latest_job('clustering', [job], self.execution, self.plan())[0], job)

    def test_finite_visible_reservation_suffix_and_exact_caps(self):
        job = T.worker_spec(1, self.execution)
        value = self.value()
        value['units'] = [dict(unit=job['unit'] + '.service', cpu_equivalents=1, memory_max_bytes=12 * T.GIB)]
        with patch.object(T, 'snapshot', return_value=value), patch.object(T.C, 'props', return_value={}), \
             patch.object(T.C, 'attest_service'):
            self.assertTrue(T.reservation_visible(job))
        value['units'][0]['memory_max_bytes'] = 11 * T.GIB
        with patch.object(T, 'snapshot', return_value=value):
            with self.assertRaises(T.C.IntegrityError):
                T.reservation_visible(job)

    def test_missing_cgroup_prevents_subsequent_admission(self):
        job = T.worker_spec(1, self.execution)
        with patch.object(T, 'snapshot', return_value=self.value()), \
             patch.object(T.C, 'props', return_value=dict(ActiveState='active')), patch.object(T.time, 'sleep'):
            with self.assertRaises(T.C.IntegrityError):
                T.reservation_visible(job)

    def test_morgan_unstarted_lanes_get_reserved_before44(self):
        self.assertEqual(T.morgan_headroom(self.value()), (16, 56 * T.GIB))
        self.put(T.ROOT43 / 'plan.json', dict(frozen=True))
        value = self.value()
        value['units'] = [dict(unit='synrfp-morgan43-e0-g0001-a.service'), dict(unit='synrfp-morgan43-m0-g0001-a.service')]
        self.assertEqual(T.morgan_headroom(value), (8, 24 * T.GIB))

    def test_completed_morgan_lane_needs_valid_binding(self):
        self.put(T.ROOT43 / 'plan.json', dict(frozen=True))
        self.put(T.ROOT43 / 'encode_worker_0_COMPLETE.json', dict(phase='complete', plan_sha256='wrong', all_shard_items_verified=True))
        with self.assertRaises(T.C.IntegrityError):
            T.morgan_headroom(self.value())

    def test_live_morgan_reservations_not_counted_twice(self):
        self.put(T.ROOT43 / 'plan.json', dict(frozen=True))
        value = self.value()
        value['units'] = [dict(unit='synrfp-morgan43-' + lane + '-g0001-a.service') for lane in T.MORGAN_LANES]
        self.assertEqual(T.morgan_headroom(value), (0, 0))

    def test_85percent_caps_and_disk_floor_unchanged(self):
        value = self.value()
        value.update(cpu_cap_sum=27, memory_cap_sum_bytes=106 * T.GIB, disk_free_bytes=39 * T.GIB)
        with patch.object(T, 'priority_context', return_value=dict(active=False)), patch.object(T, 'morgan_headroom', return_value=(0, 0)):
            reasons = T.admission(value, 1, T.GIB)
        self.assertIn('aggregate_cpu_85pct', reasons)
        self.assertIn('aggregate_memory_85pct', reasons)
        self.assertIn('disk_reserve_40GiB', reasons)

    def test_legacy41_headroom_blocks_current_extra_worker(self):
        value = self.value()
        value.update(cpu_cap_sum=18.3, memory_cap_sum_bytes=int(64.75 * T.GIB))
        with patch.object(T, 'priority_context', return_value=dict(active=True, pending_cpu=2, pending_memory_bytes=24 * T.GIB)), \
             patch.object(T, 'morgan_headroom', return_value=(0, 0)):
            self.assertIn('clustering_priority_memory_headroom', T.admission(value, 1, 12 * T.GIB))

    def test_missing_morgan_mlp_still_reserves_legacy41_gate(self):
        value = self.value()
        value.update(cpu_cap_sum=16.3, memory_cap_sum_bytes=int(48.75 * T.GIB))
        with patch.object(T, 'priority_context', return_value=dict(active=True, pending_cpu=2, pending_memory_bytes=24 * T.GIB)), \
             patch.object(T, 'morgan_headroom', return_value=(2, 16 * T.GIB)):
            reasons = T.admission(value, 1, 12 * T.GIB)
        self.assertNotIn('clustering_priority_memory_headroom', reasons)
        self.assertIn('morgan43_priority_clustering_priority_memory_headroom', reasons)

    def test_reviewed41_governor_pause_releases_future_reserve_only(self):
        with patch.object(T.P43, 'paused41', return_value=True):
            context = T.priority_context(self.value())
        self.assertFalse(context['active'])
        self.assertEqual(context['pending_memory_bytes'], 0)

    def test_reviewed_pause_keeps_85percent_caps_and_missing_morgan_reserve(self):
        value = self.value()
        value.update(cpu_cap_sum=24.5, memory_cap_sum_bytes=100 * T.GIB)
        with patch.object(T.P43, 'paused41', return_value=True), patch.object(T, 'morgan_headroom', return_value=(2, 16 * T.GIB)):
            reasons = T.admission(value, 1, 12 * T.GIB)
        self.assertIn('aggregate_memory_85pct', reasons)
        self.assertIn('morgan43_priority_aggregate_cpu_85pct', reasons)

    def test_ram_and_disk_hysteresis_preserved(self):
        value = self.value()
        value['memory_available_bytes'] = .29 * value['memory_total_bytes']
        self.assertTrue(T.pressure_state(value, True)['held'])
        value['memory_available_bytes'] = .34 * value['memory_total_bytes']
        self.assertTrue(T.pressure_state(value, True, True)['held'])
        value['memory_available_bytes'] = .36 * value['memory_total_bytes']
        self.assertFalse(T.pressure_state(value, True, True)['held'])
        value['disk_free_bytes'] = 41 * T.GIB
        self.assertTrue(T.pressure_state(value, True)['held'])
        value['disk_free_bytes'] = int(42.5 * T.GIB)
        self.assertTrue(T.pressure_state(value, True, True)['held'])
        value['disk_free_bytes'] = 44 * T.GIB
        self.assertFalse(T.pressure_state(value, True, True)['held'])

    def test_root_review_only_tested_sources_and_exact_intent(self):
        science = {str(T.ENTRY): T.B.sha(T.ENTRY)}
        report = {str(T.HERE / 'report.py'): T.B.sha(T.HERE / 'report.py')}
        controller = {str(T.HERE / 'controller.py'): T.B.sha(T.HERE / 'controller.py')}
        intent = self.put(self.root / 'INTENT.json', dict(phase='approved', policy=T.P.DECLARED_POLICY,
            config=T.P.DECLARED_CONFIG, cluster_seeds=list(range(10)), gpu_jobs=0, classification_deferred=True))
        own = self.put(self.control / 'CONTROLLER_CODE_READY.json', dict(phase='complete', tests_passed=25, sources_sha256=controller))
        scientific = self.put(self.root / 'clustering_CODE_READY.json', dict(phase='complete', tests_passed=23, sources_sha256=science))
        reports = self.put(self.root / 'report_CODE_READY.json', dict(phase='complete', tests_passed=7, sources_sha256=report))
        review = dict(schema='task-preset44-root-reviewed-v1', phase='reviewed', stages_allowed=['prepare', 'clustering'],
                      cluster_seeds=list(range(10)), declared_path_challenger=True, study43_priority_preserved=True,
                      GPU0_reserved=True, restore41_after_both_reports=True, intent_sha256=T.B.sha(intent), sources_sha256={**science, **report, **controller},
                      controller_code_ready_sha256=T.B.sha(own), clustering_code_ready_sha256=T.B.sha(scientific),
                      report_code_ready_sha256=T.B.sha(reports))
        rp = self.put(self.root / 'ROOT_REVIEW.json', review)
        with patch.object(T, 'sources', return_value=controller):
            self.assertFalse(T.manifest()['classification_dispatch'])
            review['stages_allowed'].append('mlp')
            self.put(rp, review)
            with self.assertRaises(T.C.IntegrityError):
                T.manifest()

    def test_matched_population_name_is_native_exact(self):
        # A shortened name previously failed after otherwise valid preparation.
        plan = self.plan()
        plan['tasks'][0]['population'] = 'shared_valid_all_six'
        self.assertTrue(next(iter(T.expected_paths(plan))).endswith('/shared_valid_all_six/s0.json'))

    def test_41_restore_waits_for_both_complete_reports(self):
        with patch.object(T, 'morgan_report_complete', return_value=False), patch.object(T.subprocess, 'run') as run:
            self.assertEqual(T.restore41([])['phase'], 'waiting_morgan43_complete')
            run.assert_not_called()
        with patch.object(T, 'morgan_report_complete', return_value=True), patch.object(T, 'report_complete', return_value=False), \
             patch.object(T.subprocess, 'run') as run:
            self.assertEqual(T.restore41([])['phase'], 'waiting_path44_complete')
            run.assert_not_called()

    def test_41_restore_waits_for_all_44_scientific_pids(self):
        with patch.object(T, 'morgan_report_complete', return_value=True), patch.object(T, 'report_complete', return_value=True), \
             patch.object(T.C, 'props', return_value=dict(ActiveState='inactive', MainPID='999')), patch.object(T.subprocess, 'run') as run:
            self.assertEqual(T.restore41([dict(stage='clustering', unit='owned')])['phase'], 'waiting_44_service_exit')
            run.assert_not_called()

    def test_41_restore_waits_for_43_priority_governor_after_aggregate_report(self):
        with patch.object(T, 'morgan_report_complete', return_value=True), patch.object(T, 'report_complete', return_value=True), \
             patch.object(T.C, 'props', return_value=dict(ActiveState='active', MainPID='123')), patch.object(T.subprocess, 'run') as run:
            self.assertEqual(T.restore41([])['phase'], 'waiting_43_services_and_priority_governor_exit')
            run.assert_not_called()

    def test_41_restore_waits_for_sealed_43_report_pid_even_if_state_inactive(self):
        job = dict(unit='synrfp-morgan43-r0-g0001-example', sources_sha256=self.science)
        path = self.put(T.ROOT43 / 'control/report_dispatch/r0-g0001.json', job)
        self.put(T.ROOT43 / 'control/report_dispatch_seals' / path.name, dict(sha256=T.B.sha(path)))
        def props(unit):
            return dict(ActiveState='inactive', MainPID='987' if unit == job['unit'] else '0')
        with patch.object(T.C, 'props', side_effect=props), patch.object(T.C, 'attest_service'), patch.object(T.subprocess, 'run') as run:
            self.assertFalse(T.morgan_services_idle())
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
