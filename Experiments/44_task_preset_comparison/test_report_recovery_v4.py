from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('recovery44_test', HERE / 'report_recovery_v4.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class Recovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.native = self.root / 'clustering/all_rows/schneider_ub/crxnfp__s0.json'
        self.native.parent.mkdir(parents=True)
        self.native.write_text('{}')
        self.array = self.native.with_suffix('.npz')
        self.array.write_bytes(b'preserved native assignment bytes')

    def resolve(self, name):
        return M.resolve_artifact(self.native, name, (self.root,))

    def fixture(self, artifact_name=None):
        artifact_name = artifact_name or str(self.array.relative_to(self.root))
        M.U.write(self.native, {'artifacts_sha256': {artifact_name: M.U.sha(self.array)}})
        main, export = self.root / 'main', self.root / 'export'
        plan = main / 'report/plan.json'
        M.U.write(plan, {'native_fixture': True})
        export.mkdir()
        row = dict(method='crxnfp', dataset='schneider_ub', population='all_rows', seed='0',
                   source_result_path=str(self.native), source_result_sha256=M.U.sha(self.native),
                   source_assignment_path=str(self.array), source_assignment_sha256=M.U.sha(self.array))
        table = export / 'clustering_cells.csv'
        with table.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader(); writer.writerow(row)
        M.U.write(main / 'report/COMPLETE.json', dict(phase='complete',
                  tasks={'clustering': {'complete': True, 'completed_cells': 1440}},
                  plan_sha256=M.U.sha(plan), outputs_sha256={table.name: M.U.sha(table)}))
        return main, export, row

    def test_run_root_relative_is_not_doubled(self):
        self.assertEqual(self.resolve('clustering/all_rows/schneider_ub/crxnfp__s0.npz'), self.array)

    def test_native_relative_unchanged(self):
        self.assertEqual(self.resolve(self.array.name), self.array)

    def test_absolute_unchanged(self):
        self.assertEqual(self.resolve(str(self.array)), self.array)

    def test_ambiguous_existing_names_refused(self):
        other = self.native.parent / self.array.relative_to(self.root)
        other.parent.mkdir(parents=True); other.write_bytes(self.array.read_bytes())
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.resolve(str(self.array.relative_to(self.root)))

    def test_parent_escape_refused(self):
        with self.assertRaisesRegex(ValueError, 'traversal'):
            self.resolve('../../../../outside.npz')

    def test_absolute_escape_refused(self):
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.resolve('/tmp/not-a-declared-origin.npz')

    def test_symlink_escape_refused(self):
        outside = self.root.parent / (self.root.name + '-outside')
        outside.write_bytes(b'outside'); self.addCleanup(outside.unlink)
        self.array.unlink(); self.array.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.resolve(str(self.array.relative_to(self.root)))

    def test_unknown_native_origin_refused(self):
        with self.assertRaisesRegex(ValueError, 'origin'):
            M.resolve_artifact(self.native, self.array.name, (self.root / 'other',))

    def test_all_native_rows_preserved_with_full_hash_bindings(self):
        main, export, row = self.fixture()
        before = self.native.read_bytes()
        hashes = {}
        result = M.original_rows(M.U.HashCache(), hashes, main, export, (self.root,))
        self.assertEqual(result, [row])
        self.assertEqual(hashes[str(self.array)], M.U.sha(self.array))
        self.assertEqual(self.native.read_bytes(), before)

    def test_hash_mutation_refused(self):
        main, export, _ = self.fixture()
        self.array.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            M.original_rows(M.U.HashCache(), {}, main, export, (self.root,))

    def test_assignment_binding_mutation_refused(self):
        main, export, row = self.fixture()
        row['source_assignment_sha256'] = '0' * 64
        with patch.object(M.R, 'csv_rows', return_value=[row]):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                M.original_rows(M.U.HashCache(), {}, main, export, (self.root,))

    def test_recovery_calls_unchanged_v3_and_restores_hook(self):
        original = M.V.ORIGINAL_ROWS
        def report(root):
            hashes = {}
            with patch.object(M, 'original_rows', return_value=['native rows']):
                self.assertEqual(M.V.ORIGINAL_ROWS(M.U.HashCache(), hashes), ['native rows'])
            self.assertEqual(hashes, {})
            return {'phase': 'pending'}
        with patch.object(M, 'reviewed', return_value={}), patch.object(M.V, 'once', side_effect=report):
            self.assertEqual(M.once(self.root), {'phase': 'pending'})
        self.assertIs(M.V.ORIGINAL_ROWS, original)


if __name__ == '__main__':
    unittest.main()
