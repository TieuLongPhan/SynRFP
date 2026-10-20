"""Selection controls: exact grids, family weighting and the narrow user exception."""
import copy
import csv
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('preset44_decision_tests_target', HERE / 'decision.py')
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def configs():
    out = {}
    for tokenizer, radius, sketch, mode in itertools.product(('wl', 'path', 'morgan'), (2, 3),
                                                            ('parity', 'minhash', 'cw'),
                                                            ('delta', 'union', 'delta_union', 'roles')):
        key = f'synrfp-{tokenizer}-r{radius}-b2048-{sketch}-{mode}'
        out[key] = dict(tokenizer=tokenizer, radius=radius, sketch=sketch, mode=mode,
                        n_bits=2048, kind='bits', seed=1, bbit=2,
                        count_policy='native_unweighted_support_after_full_multiset_algebra')
    return out


class DecisionTests(unittest.TestCase):
    def rows(self, task='classification'):
        rows = []
        for dataset, (config, spec), repeat in itertools.product(T.DATASETS[task], configs().items(), range(5)):
            fresh = spec['tokenizer'] == 'morgan'
            factors = {key: spec[key] for key in ('tokenizer', 'radius', 'sketch', 'mode')}
            rows.append(dict(dataset=dataset, config=config, **factors, **{'fold' if task == 'classification' else 'seed': str(repeat)},
                             mcc='0.3', ami='0.2', result_origin='fresh-study43-ECFP' if fresh else 'reused-study22-WL-Path',
                             tokenizer_profile=T.PROFILE if fresh else 'study22-original-wl-path'))
        return rows

    def plan(self):
        return dict(classification_datasets=list(T.DATASETS['classification']),
                    clustering_datasets=list(T.DATASETS['clustering']),
                    configs={key: value for key, value in configs().items() if value['tokenizer'] == 'morgan'})

    def test_exact_full_classification_and_clustering_crossings(self):
        for task, count in (('classification', 5040), ('clustering', 4320)):
            self.assertEqual(len(T.validate_cells(self.rows(task), task, self.plan(), configs())), count)

    def test_partial_grid_never_selects(self):
        with self.assertRaises(ValueError):
            T.validate_cells(self.rows()[:-1], 'classification', self.plan(), configs())

    def test_duplicate_and_foreign_cells_refused(self):
        for mutation in ('duplicate', 'foreign'):
            rows = self.rows()
            if mutation == 'duplicate':
                rows[-1] = rows[0]
            else:
                rows[0]['dataset'] = 'ehreact'
            with self.assertRaises(ValueError):
                T.validate_cells(rows, 'classification', self.plan(), configs())

    def test_old_morgan_cannot_be_relabelled_as_corrected(self):
        rows = self.rows()
        row = next(row for row in rows if row['tokenizer'] == 'morgan')
        row['result_origin'] = 'reused-study22-WL-Path'
        with self.assertRaises(ValueError):
            T.validate_cells(rows, 'classification', self.plan(), configs())

    def test_profile_and_factor_mutations_refused(self):
        for field, changed in (('tokenizer_profile', 'old-morgan'), ('radius', '4'), ('fold', '0.5'), ('mcc', 'nan')):
            rows = self.rows()
            rows[0][field] = changed
            with self.assertRaises(ValueError):
                T.validate_cells(rows, 'classification', self.plan(), configs())

    def test_full72_specs_and_corrected_subset(self):
        self.assertEqual(len(T._validate_configs(self.plan(), dict(configs=configs()))), 72)
        plan = self.plan()
        plan['configs'].pop(next(iter(plan['configs'])))
        with self.assertRaises(ValueError):
            T._validate_configs(plan, dict(configs=configs()))

    def test_family_weighting_does_not_reward_more_hierarchical_endpoints(self):
        values = {}
        for dataset in T.DATASETS['classification']:
            family = T.source_family(dataset)
            for repeat in range(5):
                # A is strong only on three-level ECReact; B on two families.
                values[dataset, 'A', repeat] = .9 if family == 'ecreact' else .1
                values[dataset, 'B', repeat] = .6 if family in ('mech', 'schneider') else .1
        ranked = T.rank(values, 'classification', {'A': {}, 'B': {}})
        self.assertEqual(ranked[0]['config'], 'B')
        self.assertAlmostEqual(ranked[0]['family_balanced_mean'], (2 * .6 + 4 * .1) / 6)

    def test_exact_ties_deterministic_and_zero_regret(self):
        values = {(d, c, repeat): .2 for d, c, repeat in itertools.product(T.DATASETS['clustering'], ('Z', 'A'), range(5))}
        ranked = T.rank(values, 'clustering', {'Z': {}, 'A': {}})
        self.assertEqual([r['config'] for r in ranked], ['A', 'Z'])
        self.assertEqual(ranked[0]['mean_family_rank'], 1.5)
        self.assertEqual(ranked[0]['mean_normalized_regret'], 0)

    def test_unknown_source_family_is_not_silently_another_family(self):
        with self.assertRaises(ValueError):
            T.source_family('new_dataset')

    def test_incomplete43_waits_without_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.assertIsNone(T.freeze('classification', path / '44', path / '43', path / 'exports', path / '22'))
            self.assertFalse((path / '44/decisions/classification.json').exists())

    def old_fixture(self, root):
        old = root / '22'
        old.mkdir()
        table = old / 'panel.csv.gz'
        table.write_bytes(b'fixed rows')
        fp = old / (T.DECLARED_CONFIG + '.npz')
        fp.write_bytes(b'packed original Path')
        meta = fp.with_suffix('.json')
        T.U.write(meta, dict(bits=2048))
        receipt = old / 'features_COMPLETE.json'
        p = dict(configs=configs(), clustering_datasets=list(T.DATASETS['clustering']),
                 panels={d: dict(table=str(table), source='source') for d in T.DATASETS['clustering']},
                 sources=dict(source=dict(feature_receipt=str(receipt), features={T.DECLARED_CONFIG: str(fp)})),
                 pinned_files={str(table): T.U.sha(table)})
        T.U.write(old / 'plan.json', p)
        T.U.write(old / 'plan_seal.json', dict(sha256=T.U.sha(old / 'plan.json')))
        T.U.write(receipt, dict(phase='complete', plan_sha256=T.U.sha(old / 'plan.json'),
                                files={fp.name: T.U.sha(fp), meta.name: T.U.sha(meta)}))
        return old, fp

    def test_user_fixed_path_exception_works_before43_and_is_not_a_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, _ = self.old_fixture(root)
            value = T.declared_path(root / '44', old)
            self.assertEqual(value['config'], T.DECLARED_CONFIG)
            self.assertFalse(value['automated_winner_selection'])
            self.assertIsNone(value['report43_complete_sha256'])
            path = root / '44/decisions/clustering_path_challenger.json'
            self.assertEqual(T.verify_decision(path), value)

    def test_fixed_decision_cannot_change_after_freeze_or_hide_cache_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, feature = self.old_fixture(root)
            value = T.declared_path(root / '44', old)
            feature.write_bytes(b'different')
            with self.assertRaises(ValueError):
                T.declared_path(root / '44', old)

    def test_declared_exception_cannot_select_morgan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, _ = self.old_fixture(root)
            value = T.declared_path(root / '44', old)
            path = root / '44/decisions/clustering_path_challenger.json'
            value['config'] = 'synrfp-morgan-r2-b2048-parity-delta'
            value['configuration']['tokenizer'] = 'morgan'
            T.U.write(path, value)
            T.U.write(path.with_name(path.stem + '_seal.json'), dict(sha256=T.U.sha(path)))
            with self.assertRaises(ValueError):
                T.verify_decision(path)

    def test_hash_cache_invalidates_same_length_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'file'
            p.write_bytes(b'aaaa')
            cache = T.U.HashCache()
            original = T.U.sha(p)
            cache.verify(p, original)
            p.write_bytes(b'bbbb')
            with self.assertRaises(ValueError):
                cache.verify(p, original)

    def report_fixture(self, root, task='clustering'):
        r43, old, exports = root / '43', root / '22', root / 'exports'
        pp = old / 'plan.json'
        T.U.write(pp, dict(configs=configs()))
        source, data = root / 'source.py', root / 'original_data'
        source.write_text('immutable source')
        data.write_bytes(b'retained original input')
        plan = dict(self.plan(), tokenizer_profile=T.PROFILE, row_cap=None, new_filtering=False,
                    source_plan_sha256=T.U.sha(pp), sources_sha256={str(source): T.U.sha(source)},
                    pinned_files={str(data): T.U.sha(data)})
        p43 = r43 / 'plan.json'
        T.U.write(p43, plan)
        directory = exports / task
        directory.mkdir(parents=True)
        rows = self.rows(task)
        cells = directory / 'cells.csv'
        with cells.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        protocol = dict(schema='morgan43-corrected72-analysis-v1', task=task, total_configs=72,
                        corrected_morgan_configs=24, reused_wl_path_configs=48, tokenizer_profile=T.PROFILE,
                        plan_sha256=T.U.sha(p43), repeats=5, row_cap=None, orders=[1, 2, 3, 4], configuration_selected=None)
        T.U.write(directory / 'protocol.json', protocol)
        inputs = {**plan['sources_sha256'], **plan['pinned_files'], str(p43): T.U.sha(p43), str(pp): T.U.sha(pp)}
        T.U.write(directory / 'inputs.json', inputs)
        outputs = {name: T.U.sha(directory / name) for name in ('cells.csv', 'protocol.json', 'inputs.json')}
        receipt = dict(schema='morgan43-report-status-v1', phase='complete', task=task, plan_sha256=T.U.sha(p43),
                       cells=len(rows), all_keyed_cells_verified=True, all_retained_artifact_hashes_verified=True,
                       source_and_feature_closure_verified=True, interaction_orders=[1, 2, 3, 4], outputs_sha256=outputs)
        T.U.write(directory / 'COMPLETE.json', receipt)
        native = dict(receipt, outputs_sha256={**inputs, **{str(directory / name): digest for name, digest in outputs.items()}})
        T.U.write(r43 / 'report' / task / 'COMPLETE.json', native)
        return r43, old, exports, source, data

    def test_complete_corrected_report_native_portable_closure_freezes_exact_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r43, old, exports, _, _ = self.report_fixture(root)
            checked = T.corrected_report('clustering', r43, exports, old)
            self.assertEqual(len(checked['values']), 4320)
            value = T.freeze('clustering', root / '44', r43, exports, old)
            self.assertEqual(value['config'], min(configs()))
            self.assertTrue(value['exploratory_selection'])
            self.assertEqual(T.freeze('clustering', root / '44', r43, exports, old), value)

    def test_complete_report_retained_native_artifact_mutation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r43, old, exports, _, artifact = self.report_fixture(root)
            artifact.write_bytes(b'changed retained artifact')
            with self.assertRaises(ValueError):
                T.corrected_report('clustering', r43, exports, old)

    def test_complete_report_stale_profile_is_not_corrected_ecfp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r43, old, exports, _, _ = self.report_fixture(root)
            plan = T.U.read(r43 / 'plan.json')
            plan['tokenizer_profile'] = 'historical-wl-like-morgan'
            T.U.write(r43 / 'plan.json', plan)
            with self.assertRaises(ValueError):
                T.corrected_report('clustering', r43, exports, old)


if __name__ == '__main__':
    unittest.main()
