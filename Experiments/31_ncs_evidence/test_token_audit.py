"""Independent algebra, collision and frozen-population boundary controls."""
from collections import Counter
import importlib.util
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ncs31_token_audit_under_test', HERE / 'token_audit.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


class AlgebraTests(unittest.TestCase):
    def test_exhaustive_small_counts_and_partial_sharing(self):
        f = A.native()
        for counts in itertools.product(range(4), repeat=4):
            r, p = Counter(dict(zip((1, 2), counts[:2]))), Counter(dict(zip((1, 2), counts[2:])))
            d, u, s = f.token_algebra(r, p)
            A.assert_algebra(r, p, d, u, s)
        r, p = Counter({1: 2}), Counter({1: 1})
        lost, gained, shared = A.occurrence_oracle(r, p)
        self.assertEqual((lost, gained, shared), (Counter({1: 1}), Counter(), Counter({1: 1})))

    def test_count_mutations_rejected(self):
        for bad in (-1, 1.0, True, '2'):
            with self.assertRaises(A.ContractError):
                A.natural({1: bad})
        for bad in (True, '1', 1.0):
            with self.assertRaises(A.ContractError):
                A.natural({bad: 1})

    def test_support_before_subtraction_mutation_rejected(self):
        r, p = Counter({1: 2}), Counter({1: 1})
        with self.assertRaises(A.ContractError):
            A.assert_algebra(r, p, Counter(), Counter({1: 3}), Counter({1: 1}))

    def test_union_shared_and_sign_mutations_rejected(self):
        r, p = Counter({1: 2}), Counter({1: 1, 2: 1})
        d, u, s = A.native().token_algebra(r, p)
        mutations = [(Counter({1: 1, 2: 1}), u, s), (d, Counter({1: 2, 2: 1}), s),
                     (d, u, Counter()), (d, Counter({1: 4, 2: 1}), s)]
        for dd, uu, ss in mutations:
            with self.assertRaises(A.ContractError):
                A.assert_algebra(r, p, dd, uu, ss)

    def test_reversal_and_original_roles_composition(self):
        import numpy as np
        f = A.native()
        r, p = Counter({10: 2, 12: 1}), Counter({10: 1, 13: 2})
        d, u, s = f.token_algebra(r, p)
        dr, ur, sr = f.token_algebra(p, r)
        self.assertEqual(dict(dr), {k: -v for k, v in d.items()})
        self.assertEqual((u, s), (ur, sr))
        out = f.encode_tokens(r, p, 'parity')
        signed = f._packed(r, p, 'parity', 1024, 'delta')
        shared = f._packed(Counter(), Counter({k: 1 for k in s}), 'parity', 1024, 'union')
        np.testing.assert_array_equal(out['roles'], np.concatenate([signed, shared]))


class CollisionTests(unittest.TestCase):
    def test_forced_sketch_collision_and_support_loss_are_separate(self):
        rows = [(((1, 1),), (1,), b'X'), (((1, 2),), (1,), b'X'),
                (((2, 1),), (2,), b'X')]
        result = A.equality_counts(rows)
        self.assertEqual(result['equal_full_count_pairs'], 0)
        self.assertEqual(result['equal_support_pairs'], 1)
        self.assertEqual(result['equal_sketch_pairs'], 3)
        self.assertEqual(result['equal_sketch_different_support_pairs'], 2)

    def test_digest_collision_cannot_force_full_equality(self):
        with patch.object(A, 'digest', return_value='forced-collision'):
            result = A.equality_counts([(((1, 1),), (1,), b'A'), (((2, 1),), (2,), b'B')])
        self.assertEqual(result['equal_full_count_pairs'], 0)
        self.assertEqual(result['equal_sketch_pairs'], 0)

    def test_same_support_different_output_is_a_hard_failure(self):
        with self.assertRaises(A.ContractError):
            A.equality_counts([(((1, 1),), (1,), b'A'), (((1, 2),), (1,), b'B')])

    def test_native_graph_renumbering_invariance(self):
        import numpy as np
        f = A.native()
        rxn = A.reaction('[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]', 'classification')
        from synrfp.graph.molecule import Molecule
        def permute(graph):
            maps = {node: 100 - 3 * index for index, node in enumerate(reversed(list(graph.nodes)))}
            nodes = {maps[node]: dict(attrs) for node, attrs in reversed(list(graph.nodes.items()))}
            edges = {(maps[u], maps[v]): dict(attrs) for (u, v), attrs in reversed(list(graph.edges.items()))}
            return Molecule.from_dicts(nodes, edges)
        for name, radius in A.SETTINGS:
            tok = f._tokenizer(name, f.NODE_ATTRS, f.EDGE_ATTRS)
            r, p = tok.tokens_graph(rxn.reactant, radius), tok.tokens_graph(rxn.product, radius)
            rr, pp = tok.tokens_graph(permute(rxn.reactant), radius), tok.tokens_graph(permute(rxn.product), radius)
            self.assertEqual((r, p), (rr, pp))
            for mode in A.MODES:
                np.testing.assert_array_equal(f.encode_tokens(r, p, 'parity')[mode],
                                              f.encode_tokens(rr, pp, 'parity')[mode])


class PopulationTests(unittest.TestCase):
    def test_producer_receipt_binds_attempt_and_rejects_unsafe_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'token_audit'
            root.mkdir()
            A.write(root / 'COMPLETE.json', {'schema': A.SCHEMA, 'phase': 'complete', 'outputs': {}})
            stage = '0' * 32
            with patch.dict(os.environ, {'NCS_EVIDENCE_STAGE_ID': stage, 'NCS_EVIDENCE_INPUT_KEY': 'test-input'}):
                A.producer_ready(root)
            receipt = json.loads((root.parent / 'stages' / stage / 'PRODUCER_READY.json').read_text())
            self.assertEqual(receipt['stage_id'], stage)
            self.assertEqual(receipt['input_key'], 'test-input')
            self.assertEqual(receipt['result_sha256'], A.sha(root / 'COMPLETE.json'))
            with patch.dict(os.environ, {'NCS_EVIDENCE_STAGE_ID': '../escape', 'NCS_EVIDENCE_INPUT_KEY': 'test-input'}):
                with self.assertRaises(A.ContractError):
                    A.producer_ready(root)

    def test_sampling_is_label_blind_and_retains_ineligible(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.csv'
            frame = pd.DataFrame({'row_id': ['a', 'b', 'c'], 'left': ['C', 'CC', 'CCC'],
                                  'right': ['C', 'C=C', 'CC=C'], 'eligible': ['True', 'False', 'True'],
                                  'target_secret': [3, 2, 1]})
            frame.to_csv(path, index=False)
            source = {'table': str(path), 'n': 3, 'input_columns': {'left': 'left', 'right': 'right', 'eligible': 'eligible'}}
            first = A.select_sample(source, 'pool', 'source', sample_rows=3)
            frame['target_secret'] = [999, 0, 37]
            frame.to_csv(path, index=False)
            second = A.select_sample(source, 'pool', 'source', sample_rows=3)
            self.assertEqual(first, second)
            self.assertEqual([x[1] for x in first], [0, 1, 2])
            self.assertFalse(first[1][-1])
            with patch.object(pd, 'read_csv', wraps=pd.read_csv) as reader:
                A.select_sample(source, 'pool', 'source', sample_rows=1)
            data_calls = [call for call in reader.call_args_list if 'usecols' in call.kwargs]
            self.assertEqual(len(data_calls), 1)
            self.assertNotIn('target_secret', data_calls[0].kwargs['usecols'])

    def test_missing_population_and_label_column_are_refused(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.csv'
            pd.DataFrame({'row_id': ['a'], 'rxn': ['C>>C'], 'target': [1]}).to_csv(path, index=False)
            with self.assertRaises(A.ContractError):
                A.input_columns({'table': str(path), 'input_columns': {'reaction': 'target'}})
            with self.assertRaises(A.ContractError):
                A.select_sample({'table': str(path), 'n': 2}, 'pool', 'key')

    def test_chemical_parse_refusal_and_algorithm_exception_boundary(self):
        with self.assertRaises(ValueError):
            A.reaction('invalid>>C', 'yield')
        with self.assertRaises(A.ContractError):
            A.reaction('C>>C', 'unknown-profile')
        with self.assertRaises(RuntimeError):
            with patch.object(A.native().Reaction, 'from_rsmi', side_effect=RuntimeError('algorithm failed')):
                A.reaction('C>>C', 'classification')

    def test_source_receipts_preserve_refusals_without_raw_payload(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'source.csv'
            pd.DataFrame({'row_id': ['ok', 'bad', 'ineligible'],
                          'left': ['CCO', 'not-a-molecule', 'C'],
                          'right': ['CC=O', 'C', 'C'],
                          'eligible': ['True', 'True', 'False'],
                          'target_secret': [123, 456, 789]}).to_csv(path, index=False)
            source = {'table': str(path), 'n': 3, 'parser_profile': 'yield',
                      'input_columns': {'left': 'left', 'right': 'right', 'eligible': 'eligible'}}
            item = {'pool': 'fixture', 'key': 'fixture', 'source': source,
                    'table_sha256': A.sha(path), 'aliases': ['fixture/fixture']}
            receipt = A.audit_source(item, root / 'run', 'test-pins')
            self.assertEqual(receipt['sampled_rows'], 3)
            self.assertEqual(receipt['successful_setting_rows'], 6)
            self.assertEqual(receipt['refusals'], {'source_parse': 1, 'original_eligibility': 1})
            rowfile = next((root / 'run').rglob('rows.jsonl'))
            rows = [json.loads(line) for line in rowfile.read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertNotIn('not-a-molecule', rowfile.read_text())
            self.assertNotIn('target_secret', rowfile.read_text())
            with patch.object(A, 'reaction', side_effect=RuntimeError('algorithm defect')):
                with self.assertRaises(RuntimeError):
                    A.audit_source(item, root / 'failed-run', 'test-pins')
            self.assertFalse(list((root / 'failed-run').rglob('COMPLETE.json')))


if __name__ == '__main__':
    unittest.main()
