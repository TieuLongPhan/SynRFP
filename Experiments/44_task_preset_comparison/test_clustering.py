"""Independent identity, population, scientific and cooperative-worker controls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('task44_clustering_under_test', HERE / 'clustering.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class IdentityTests(unittest.TestCase):
    def panels(self):
        shared = {'mech_1', 'mech_2', 'schneider_ub', 'uspto50k_ub'}
        return {d: dict(n_rows=100 + i, shared_is_all=d in shared) for i, d in enumerate(C.ENDPOINTS)}

    def test_exact_ten_seed_grid_and_populations(self):
        tasks = C.expected_tasks(self.panels())
        self.assertEqual(len(tasks), 240)
        self.assertEqual(len({C.task_key(t) for t in tasks}), 240)
        self.assertEqual(sum(t['origin'] == 'fresh' for t in tasks), 200)
        self.assertEqual(sum(t['origin'] == 'population_identical_copy' for t in tasks), 40)
        self.assertEqual({t['seed'] for t in tasks}, set(range(10)))
        self.assertEqual(tasks, C.expected_tasks({'panels': self.panels()}))

    def test_one_and_two_worker_shards_keep_copy_dependencies(self):
        p = {'tasks': C.expected_tasks(self.panels())}
        self.assertEqual(C.stage_items(p, 0, 1), p['tasks'])
        first, second = C.stage_items(p, 0, 2), C.stage_items(p, 1, 2)
        self.assertEqual(len(first), len(second))
        self.assertFalse({C.task_key(t) for t in first} & {C.task_key(t) for t in second})
        self.assertEqual(len(first) + len(second), 240)
        for shard in (first, second):
            for i in range(0, len(shard), 2):
                self.assertEqual([t['population'] for t in shard[i:i+2]], C.POLICIES)
                self.assertEqual(shard[i]['seed'], shard[i+1]['seed'])

    def test_split_population_pair_is_refused(self):
        p = {'tasks': C.expected_tasks(self.panels())}
        p['tasks'][0], p['tasks'][2] = p['tasks'][2], p['tasks'][0]
        with self.assertRaisesRegex(ValueError, 'pair'):
            C.stage_items(p, 0, 2)

    def test_environment_drift_is_refused(self):
        bad = C.environment(); bad['scikit-learn'] = 'another-build'
        with self.assertRaisesRegex(ValueError, 'original synrfp'):
            C.check_environment(bad)

    def test_hash_cache_detects_same_size_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'source'; p.write_text('old')
            cache = C.HashCache(); expected = C.sha(p); cache.verify(p, expected)
            p.write_text('new')
            with self.assertRaisesRegex(ValueError, 'Changed pinned'):
                cache.verify(p, expected)

    def test_immutable_receipt_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'COMPLETE.json'; C.immutable(p, {'phase': 'complete'})
            with self.assertRaisesRegex(ValueError, 'already retained'):
                C.immutable(p, {'phase': 'yielded'})
            self.assertEqual(C.read(p), {'phase': 'complete'})

    def test_preparation_recovers_after_verified_plan_before_ready_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); (root / 'clustering').mkdir()
            decision = root / 'decisions/clustering_path_challenger.json'
            plan = dict(decision_path=str(decision), decision_sha256='sealed-decision', sources_sha256={'source': 'sealed-source'})
            C.write(C.plan_path(root), plan)
            C.write(C.stage_root(root) / 'plan_seal.json', {'sha256': C.sha(C.plan_path(root))})
            marker = C.stage_root(root) / 'PLAN_READY.json'
            self.assertFalse(marker.exists())
            with patch.object(C, 'verify', return_value=plan) as checked, patch.object(C, 'native') as no_new_prepare:
                self.assertEqual(C.prepare(root, decision), plan)
                checked.assert_called_once_with(root)
                no_new_prepare.assert_not_called()
            first = marker.read_bytes(); receipt = C.read(marker)
            self.assertEqual(receipt['plan_sha256'], C.sha(C.plan_path(root)))
            self.assertEqual((receipt['expected_items'], receipt['fresh_fits'], receipt['population_copies']), (240, 200, 40))
            with patch.object(C, 'verify', return_value=plan):
                C.prepare(root, decision)
            self.assertEqual(marker.read_bytes(), first)

    def test_preparation_does_not_overwrite_conflicting_ready_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); (root / 'clustering').mkdir()
            decision = root / 'decision.json'
            plan = dict(decision_path=str(decision), decision_sha256='sealed-decision', sources_sha256={})
            C.write(C.plan_path(root), plan)
            marker = C.stage_root(root) / 'PLAN_READY.json'
            C.write(marker, {'phase': 'complete', 'plan_sha256': 'foreign-plan'})
            original = marker.read_bytes()
            with patch.object(C, 'verify', return_value=plan):
                with self.assertRaisesRegex(ValueError, 'closure differs'):
                    C.prepare(root, decision)
            self.assertEqual(marker.read_bytes(), original)

    def test_source_ids_text_positions_and_validity_all_required(self):
        table = pd.DataFrame(dict(row_id=['a', 'b'], source_position=[0, 1], rxn=['C>>C', 'N>>N']))
        rt = pd.DataFrame(dict(r_id=['a', 'b'], rxn=['C>>C', 'N>>N']))
        features = dict(row_ids=np.array(['a', 'b']), source_positions=np.arange(2), ok=np.array([True, False]))
        anchor = type('Anchor', (), {'ok': np.array([True, False])})()
        C.compare_source(table, rt, features, anchor)
        mutations = [table.assign(row_id=['b', 'a']), table.assign(source_position=[1, 0]), table.assign(rxn=['N>>N', 'C>>C'])]
        for bad in mutations:
            with self.assertRaises(ValueError):
                C.compare_source(bad, rt, features, anchor)
        with self.assertRaisesRegex(ValueError, 'validity'):
            C.compare_source(table, rt, features, type('Anchor', (), {'ok': np.array([True, True])})())

    def test_endpoint_alignment_and_labels_are_required(self):
        table = pd.DataFrame(dict(row_id=['a', 'b'], source_position=[2, 4], y=[0, 1]))
        ds = pd.DataFrame(dict(r_id=['a', 'b'], row=[2, 4], y=[0, 1]))
        C.compare_panel(table, ds)
        for bad in [table.iloc[::-1], table.assign(y=[1, 0]), table.assign(source_position=[4, 2])]:
            with self.assertRaises(ValueError):
                C.compare_panel(bad, ds)

    def test_wrong_invalid_zero_row_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'features.npz'
            X = np.zeros((2, 256), np.uint8); X[1, 0] = 1
            C.save_npz(p, X=X, ok=np.array([True, False]), n_bits=2048, source_positions=np.arange(2), row_ids=np.array(['a', 'b']))
            with self.assertRaisesRegex(ValueError, 'remain zero'):
                C.source_arrays(p, 2)

    def test_both_filesystem_disk_floors(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder); (p / 'clustering').mkdir()
            seen = []
            def usage(path):
                seen.append(path)
                return type('Usage', (), {'free': 39 * 1024**3 if len(seen) == 2 else 41 * 1024**3})()
            with patch.object(C.shutil, 'disk_usage', side_effect=usage):
                with self.assertRaisesRegex(ValueError, '40GiB'):
                    C.disk_guard(p)
            self.assertEqual(len(seen), 2)

    def test_only_declared_unchanged_path_can_bypass_corrected_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = root / 'old'; old.mkdir()
            config = 'synrfp-path-r2-b2048-parity-delta'
            cfg = dict(tokenizer='path', radius=2, sketch='parity', mode='delta', n_bits=2048, seed=1)
            C.write(old / 'plan.json', {'configs': {config: cfg}})
            path = root / 'clustering_path_challenger.json'
            d = dict(phase='frozen', task='clustering', policy='user_declared_path_challenger_20261020', config=config,
                     configuration=cfg, tokenizer_profile='study22-original-wl-path', source_plan_sha256=C.sha(old / 'plan.json'),
                     sources_sha256={}, candidate_status='provisional_user_requested_challenger', auto_winner=False,
                     plan43_sha256=None, report43_complete_sha256=None)
            C.write(path, d); C.write(path.with_name(path.stem + '_seal.json'), {'sha256': C.sha(path)})
            with patch.object(C, 'ROOT22', old):
                value, report, _ = C.verify_decision(path)
                self.assertEqual(value['config'], config); self.assertIsNone(report)
                for change in [{'auto_winner': True}, {'tokenizer_profile': C.PROFILE}, {'config': 'synrfp-morgan-r2-b2048-parity-delta'}, {'candidate_status': 'winner'}]:
                    C.write(path, {**d, **change})
                    with self.assertRaises(ValueError):
                        C.verify_decision(path)


class ActualFitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / 'clustering').mkdir()
        self.ds = pd.DataFrame(dict(row=np.arange(12), r_id=[f'r{i}' for i in range(12)],
                                    y=np.repeat([0, 1], 6), label=np.repeat([0, 1], 6)))
        table = self.root / 'dataset.pkl'; self.ds.to_pickle(table)
        X = np.zeros((12, 2048), np.uint8); X[6:, :50] = 1; X[np.arange(12), np.arange(12) + 100] = 1
        self.X = X.astype(np.float32)
        fp = self.root / 'features.npz'
        C.save_npz(fp, X=np.packbits(X, axis=1), ok=np.ones(12, bool), n_bits=2048,
                   source_positions=np.arange(12), row_ids=self.ds.r_id.to_numpy(dtype=str))
        population = self.root / 'population.npz'
        C.save_npz(population, all_rows=np.ones(12, bool), shared_valid=np.ones(12, bool), valid=np.ones(12, bool), source_row=np.arange(12))
        self.panel = dict(source='fixture', table=str(table), dataset_sha256=C.sha(table), n_rows=12, k=2,
                          params=C.native().NATIVE.SMALL_PARAMS, method='kmeans', population_file=str(population),
                          shared_is_all=True, label='fixture_label', spec={'label': 'fixture_label'})
        self.tasks = [dict(dataset='mech_2', population=p, seed=0,
                           origin='fresh' if p == 'all_rows' else 'population_identical_copy') for p in C.POLICIES]
        self.plan = dict(panels={'mech_2': self.panel}, sources={'fixture': {'path': str(fp), 'sha256': C.sha(fp), 'rows': 12}},
                         config='synrfp-path-r2-b2048-parity-delta', configuration={'tokenizer': 'path', 'n_bits': 2048, 'seed': 1},
                         tokenizer_profile='study22-original-wl-path', decision_sha256='fixture-decision',
                         protocol='known_k_10_seeds', tasks=self.tasks, pins={str(population.resolve()): C.sha(population)},
                         sources_sha256={}, decision_policy='user_declared_path_challenger_20261020')
        C.write(C.plan_path(self.root), self.plan)
        self.env = patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}); self.env.start()
        self.guard = patch.object(C, 'disk_guard'); self.guard.start()
        self.review = patch.object(C, 'review'); self.review.start()

    def tearDown(self):
        self.review.stop(); self.guard.stop(); self.env.stop(); self.temp.cleanup()

    def test_native_recipe_actual_fit_full_saved_center_and_metric_replay(self):
        r = C.fit(self.root, self.plan, self.tasks[0], threads=1)
        self.assertEqual(r['identity']['params'], C.native().NATIVE.SMALL_PARAMS)
        self.assertTrue(r['prediction_replay']); self.assertTrue(r['full_metric_replay'])
        self.assertEqual(r['n_scored'], 12)
        self.assertEqual(r['scores']['fixture_label']['ami'], 1.)
        replay = C.verify_result(self.root, self.plan, self.tasks[0], deep=True)
        self.assertEqual(replay['artifacts_sha256'], r['artifacts_sha256'])
        self.assertEqual(C.fit(self.root, self.plan, self.tasks[0], threads=1), r)

    def test_population_copy_reuses_verified_artifacts(self):
        first = C.fit(self.root, self.plan, self.tasks[0], threads=1)
        copied = C.fit(self.root, self.plan, self.tasks[1], threads=1)
        self.assertEqual(copied['fit_seconds'], 0.)
        self.assertEqual(copied['artifacts_sha256'], first['artifacts_sha256'])
        self.assertEqual(copied['copied_result_sha256'], C.sha(C.result_path(self.root, self.tasks[0])))
        C.verify_result(self.root, self.plan, self.tasks[1], deep=True)

    def test_copy_cannot_precede_original_fit(self):
        with self.assertRaisesRegex(ValueError, 'source must close first'):
            C.fit(self.root, self.plan, self.tasks[1], threads=1)

    def test_mutated_metric_refused_by_independent_replay(self):
        r = C.fit(self.root, self.plan, self.tasks[0], threads=1)
        bad = copy.deepcopy(r); bad['scores']['fixture_label']['ami'] = -.5
        with self.assertRaisesRegex(ValueError, 'metric replay'):
            C.verify_result(self.root, self.plan, self.tasks[0], record=bad, deep=True)

    def test_mutated_row_identity_is_refused_even_with_new_digest(self):
        r = C.fit(self.root, self.plan, self.tasks[0], threads=1)
        ap = Path(r['assignment_path'])
        with np.load(ap, allow_pickle=False) as z:
            values = {k: z[k].copy() for k in z.files}
        values['r_id'] = values['r_id'][::-1]
        C.save_npz(ap, **values); r['artifacts_sha256'][str(ap)] = C.sha(ap)
        with self.assertRaisesRegex(ValueError, 'identity order'):
            C.verify_result(self.root, self.plan, self.tasks[0], record=r)

    def test_mutated_saved_centers_fail_native_full_prediction(self):
        r = C.fit(self.root, self.plan, self.tasks[0], threads=1)
        cp = Path(r['centers_path']); centers = np.load(cp, allow_pickle=False)[::-1].copy()
        with cp.open('wb') as f:
            np.save(f, centers, allow_pickle=False)
        r['artifacts_sha256'][str(cp)] = C.sha(cp)
        with self.assertRaisesRegex(ValueError, 'full prediction replay'):
            C.verify_result(self.root, self.plan, self.tasks[0], record=r, deep=True)

    def test_false_identity_and_historical_morgan_are_not_relabelled(self):
        r = C.fit(self.root, self.plan, self.tasks[0], threads=1)
        r['identity']['tokenizer_profile'] = 'historical-morgan'
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            C.verify_result(self.root, self.plan, self.tasks[0], record=r)

    def test_secondary_numerical_oracle_allows_exact_tie_and_rejects_false_merge(self):
        zeros = np.zeros((2, 2048), np.float32)
        result = C.independent_center_check(zeros[:1], zeros, np.array([1]))
        self.assertEqual(result['mismatches'], 1)
        centers = zeros.copy(); centers[1] = 1
        with self.assertRaisesRegex(ValueError, 'forward bound'):
            C.independent_center_check(zeros[:1], centers, np.array([1]))

    def test_cuda_visibility_is_rejected_before_fit(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(ValueError, 'hide CUDA'):
                C.fit(self.root, self.plan, self.tasks[0], threads=1)


class WorkerTests(unittest.TestCase):
    def test_signal_sets_cooperative_flag(self):
        C.YIELD = False; C.request_yield(None, None); self.assertTrue(C.YIELD)

    def test_review_explicit_path_exception_and_source_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source.py'; source.write_text('unchanged')
            p = dict(sources_sha256={str(source): C.sha(source)}, decision_policy='user_declared_path_challenger_20261020',
                     config='synrfp-path-r2-b2048-parity-delta')
            r = dict(schema='task-preset44-root-reviewed-v1', phase='reviewed', stages_allowed=['prepare', 'clustering'],
                     cluster_seeds=list(range(10)), study43_priority_preserved=True, GPU0_reserved=True,
                     declared_path_challenger=True, task43_report_prerequisite=False, sources_sha256=p['sources_sha256'])
            C.write(root / 'ROOT_REVIEW.json', r); C.review(root, p)
            for changed in [{'declared_path_challenger': False}, {'GPU0_reserved': False}, {'cluster_seeds': list(range(25))}]:
                C.write(root / 'ROOT_REVIEW.json', {**r, **changed})
                with self.assertRaises(ValueError):
                    C.review(root, p)
            C.write(root / 'ROOT_REVIEW.json', r); source.write_text('mutated')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                C.review(root, p)

    def test_one_cpu_worker_yields_only_after_a_closed_cell_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'clustering').mkdir()
            tasks = [dict(dataset='mech_2', population=p, seed=0,
                          origin='fresh' if p == 'all_rows' else 'population_identical_copy') for p in C.POLICIES]
            plan = dict(tasks=tasks, sources_sha256={}, decision_sha256='decision')
            C.write(C.plan_path(root), plan)
            closed = {}
            def checked(root_, plan_, task, **kwargs):
                return closed.get(C.task_key(task))
            def finish(root_, plan_, task, threads):
                value = {'phase': 'complete'}
                closed[C.task_key(task)] = value
                C.write(C.result_path(root_, task), value)
                C.request_yield(None, None)
                return value
            with patch.object(C, 'verify', return_value=plan), patch.object(C, 'review'), \
                 patch.object(C, 'verify_result', side_effect=checked), patch.object(C, 'fit', side_effect=finish), \
                 patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
                first = C.run(root, worker=0, workers=1, threads=1, run_id='generation-1')
                self.assertEqual(first['phase'], 'yielded'); self.assertEqual(first['audited_items'], 1)
                self.assertTrue((root / 'workers/generation-1/STARTED.json').exists())
                self.assertTrue((root / 'workers/generation-1/YIELDED.json').exists())
                self.assertFalse((root / 'clustering_worker_0_COMPLETE.json').exists())
                second = C.run(root, worker=0, workers=1, threads=1, run_id='generation-2')
                self.assertEqual(second['phase'], 'complete'); self.assertEqual(second['audited_items'], 2)
                self.assertEqual(second['expected_items'], 2)
                self.assertTrue(second['all_shard_items_verified'])
                self.assertEqual(set(second['outputs_sha256']), {str(C.result_path(root, t).resolve()) for t in tasks})
                self.assertTrue((root / 'workers/generation-2/COMPLETE.json').exists())
                self.assertEqual(C.read(root / 'clustering_worker_0_COMPLETE.json'), second)
                with self.assertRaisesRegex(ValueError, 'already retained'):
                    C.run(root, worker=0, workers=1, threads=1, run_id='generation-1')


if __name__ == '__main__':
    unittest.main()
