"""Five-shard exact-key/provenance mutations without fitting or service actions."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('five_shard_report_under_test', HERE / 'report_parallel_v3.py')
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
U = T.U


class FiveShardReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        shared = {'mech_1', 'mech_2', 'schneider_ub', 'uspto50k_ub'}
        self.tasks = [dict(dataset=dataset, population=population, seed=seed,
                           origin='population_identical_copy' if population != 'all_rows' and dataset in shared else 'fresh')
                      for dataset in T.R.ENDPOINTS for seed in range(10) for population in T.R.POLICIES]
        source = self.root / 'native-science.py'; source.write_text('native-unchanged')
        self.scientific = dict(tasks=self.tasks, sources_sha256={str(source): U.sha(source)})
        self.scientific_path = self.root / 'clustering/plan.json'; U.write(self.scientific_path, self.scientific)
        assignments = {str(worker): [] for worker in range(5)}
        for i in range(0, len(self.tasks), 2):
            assignments[str((i // 2) % 5)].extend(self.tasks[i:i+2])
        self.op = dict(schema='task-preset44-five-shard-plan-v3', scientific_plan_sha256=U.sha(self.scientific_path),
                       workers=5, threads=1, memory_gib=7, assignments=assignments,
                       sources_sha256=T.source_pins(), inputs_sha256={str(self.scientific_path): U.sha(self.scientific_path)})
        self.oproot = self.root / 'parallel_v3'
        U.write(self.oproot / 'plan.json', self.op)
        U.write(self.oproot / 'plan_seal.json', {'sha256': U.sha(self.oproot / 'plan.json')})
        U.write(self.oproot / 'CODE_READY.json', dict(phase='complete', tests_passed=10, sources_sha256=self.op['sources_sha256']))
        self.review = dict(schema='task-preset44-five-shard-root-reviewed', phase='reviewed', workers=5,
            worker_cpu=1, worker_threads=1, worker_memory_gib=7, gpu_jobs=0,
            plan_sha256=U.sha(self.oproot / 'plan.json'), code_ready_sha256=U.sha(self.oproot / 'CODE_READY.json'),
            sources_sha256={**self.scientific['sources_sha256'], **self.op['sources_sha256']})
        U.write(self.oproot / 'ROOT_REVIEW.json', self.review)
        self.producer = SimpleNamespace(plan_path=lambda root: Path(root) / 'clustering/plan.json',
            result_path=lambda root, task: Path(root) / 'clustering/cells' / task['dataset'] / task['population'] / ('s' + str(task['seed']) + '.json'),
            expected_tasks=lambda plan: plan['tasks'], verify=lambda root: self.scientific)
        for worker in range(5):
            self.make_shard(worker)
        current = patch.object(T, 'parallel', return_value=SimpleNamespace(verify_partition=lambda root: self.op))
        current.start(); self.addCleanup(current.stop)

    def make_shard(self, worker):
        outputs = {}
        for task in self.op['assignments'][str(worker)]:
            path = self.producer.result_path(self.root, task)
            # Existing five-thread cells are valid retained scientific evidence.
            U.write(path, dict(phase='complete', threads=5 if task['seed'] == 0 else 1, value=1))
            outputs[str(path.resolve())] = U.sha(path)
        run_id = 'v3-w' + str(worker) + '-g1'; invocation = self.root / 'workers' / run_id
        base = dict(stage='clustering', worker=worker, workers=5, run_id=run_id,
                    plan_sha256=U.sha(self.scientific_path), sources_sha256=self.scientific['sources_sha256'])
        U.write(invocation / 'STARTED.json', dict(**base, phase='ready', pid=1000+worker))
        terminal = dict(**base, phase='complete', expected_items=len(outputs), audited_items=len(outputs),
                        all_shard_items_verified=True, outputs_sha256=outputs)
        U.write(invocation / 'COMPLETE.json', terminal)
        U.write(self.root / ('clustering_worker_' + str(worker) + '_COMPLETE.json'), terminal)
        U.write(invocation / 'OPERATIONAL.json', dict(phase='ready', schema='task-preset44-five-shard-worker-v3',
            run_id=run_id, worker=worker, workers=5, threads=1, memory_gib=7, pid=1000+worker,
            partition_plan_sha256=U.sha(self.oproot / 'plan.json'), scientific_plan_sha256=U.sha(self.scientific_path),
            sources_sha256=self.op['sources_sha256'], outputs_sha256={str((self.oproot / name).resolve()): U.sha(self.oproot / name)
                for name in ('plan.json', 'plan_seal.json', 'ROOT_REVIEW.json', 'CODE_READY.json')}))

    def closed(self):
        hashes = {}
        result = T.shard_closed(self.root, self.scientific, hashes, U.HashCache(), self.producer)
        return result, hashes

    def test_five_shards_close_exact240_cells_and_bind_operations(self):
        result, hashes = self.closed()
        self.assertTrue(result)
        self.assertEqual(len({str(self.producer.result_path(self.root, t)) for t in self.tasks}), 240)
        for worker in range(5):
            self.assertIn(str(self.root / f'workers/v3-w{worker}-g1/OPERATIONAL.json'), hashes)
        for name in ('plan.json', 'plan_seal.json', 'ROOT_REVIEW.json', 'CODE_READY.json'):
            self.assertIn(str(self.oproot / name), hashes)

    def test_missing_shard_or_unclosed_worker_ops_waits(self):
        path = self.root / 'clustering_worker_4_COMPLETE.json'; saved = path.read_bytes(); path.unlink()
        self.assertFalse(self.closed()[0]); path.write_bytes(saved)
        (self.root / 'workers/v3-w4-g1/OPERATIONAL.json').unlink()
        self.assertFalse(self.closed()[0])

    def test_old_one_or_two_worker_cohort_cannot_close_five(self):
        for workers in (1, 2):
            path = self.root / 'clustering_worker_0_COMPLETE.json'; r = U.read(path); r['workers'] = workers; U.write(path, r)
            with self.assertRaisesRegex(ValueError, 'five-shard'):
                self.closed()

    def test_wrong_shard_membership_rejected_even_with_total_count(self):
        path = self.root / 'clustering_worker_0_COMPLETE.json'; r = U.read(path)
        first = next(iter(r['outputs_sha256'])); r['outputs_sha256'].pop(first)
        other = U.read(self.root / 'clustering_worker_1_COMPLETE.json')['outputs_sha256']; foreign = next(iter(other))
        r['outputs_sha256'][foreign] = other[foreign]; U.write(path, r)
        with self.assertRaisesRegex(ValueError, 'keyed'):
            self.closed()

    def test_split_pair_or_changed_scientific_plan_is_rejected(self):
        original = copy.deepcopy(self.op)
        self.op['assignments']['0'][0], self.op['assignments']['1'][0] = self.op['assignments']['1'][0], self.op['assignments']['0'][0]
        with self.assertRaisesRegex(ValueError, 'dependencies'):
            self.closed()
        self.op = original; self.op['scientific_plan_sha256'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'scientific plan'):
            self.closed()

    def test_modified_result_digest_is_refused(self):
        path = self.producer.result_path(self.root, self.tasks[0]); r = U.read(path); r['value'] = 2; U.write(path, r)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.closed()

    def test_wrong_ops_pid_thread_or_partition_refused(self):
        path = self.root / 'workers/v3-w0-g1/OPERATIONAL.json'; original = U.read(path)
        for field, value in (('pid', 9), ('threads', 5), ('partition_plan_sha256', 'foreign')):
            U.write(path, {**original, field: value})
            with self.assertRaisesRegex(ValueError, 'operational witness'):
                self.closed()

    def test_run_and_aggregate_terminal_must_agree(self):
        path = self.root / 'workers/v3-w0-g1/COMPLETE.json'; r = U.read(path); r['audited_items'] -= 1; U.write(path, r)
        with self.assertRaisesRegex(ValueError, 'receipts disagree'):
            self.closed()

    def test_changed_resource_profile_or_unreviewed_adapter_refused(self):
        path = self.oproot / 'ROOT_REVIEW.json'
        for field, value in (('worker_memory_gib', 12), ('worker_cpu', 2), ('sources_sha256', self.scientific['sources_sha256'])):
            U.write(path, {**self.review, field: value})
            with self.assertRaises(ValueError):
                self.closed()

    def test_original_reporting_math_restored_after_provenance_adapter(self):
        old_shards, old_rows = T.R.shard_closed, T.R.original_rows
        def original_once(root):
            hashes = {}; cache = U.HashCache()
            self.assertTrue(T.R.shard_closed(root, self.scientific, hashes, cache, self.producer))
            self.assertEqual(T.R.original_rows(cache, hashes), ['original-baseline-rows'])
            self.assertIn(str(self.oproot / 'ROOT_REVIEW.json'), hashes)
            return {'phase': 'pending'}
        with patch.object(U, 'module', return_value=self.producer), patch.object(T.R, 'once', side_effect=original_once), \
             patch.object(T, 'ORIGINAL_ROWS', return_value=['original-baseline-rows']):
            self.assertEqual(T.once(self.root)['phase'], 'pending')
        self.assertIs(T.R.shard_closed, old_shards)
        self.assertIs(T.R.original_rows, old_rows)


if __name__ == '__main__':
    unittest.main()
