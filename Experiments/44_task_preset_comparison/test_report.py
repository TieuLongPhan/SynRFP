import itertools
from pathlib import Path
import tempfile
import unittest

import common as U
import report as R


def fixture():
    original, candidate = [], []
    for population, seed in itertools.product(R.POLICIES, range(10)):
        for method in (*R.BASELINES, 'synrfp_selected'):
            original.append(dict(dataset='toy', population=population, seed=seed, method=method,
                                 configuration=R.SHARED_CONFIG if method == 'synrfp_selected' else method,
                                 n_rows=30, n_scored=30, k=3, ami=.2, ari=.1, nmi=.3, mcc_hungarian=.4))
        candidate.append(dict(dataset='toy', population=population, seed=seed, method='path',
                              configuration='path', n_rows=30, n_scored=30, k=3,
                              ami=.3 + seed*.001, ari=.2, nmi=.4, mcc_hungarian=.5))
    return original, candidate


class ReportTests(unittest.TestCase):
    def test_shared_table_retained_as_seventh_arm(self):
        old, new = fixture()
        primary, both = R.combine(old, new, 'path', endpoints=('toy',))
        self.assertEqual((len(primary), len(both)), (120, 140))
        shared = [r for r in both if r['method'] == 'synrfp_shared']
        self.assertTrue(all(r['ami'] == .2 and r['configuration'] == R.SHARED_CONFIG for r in shared))
        self.assertEqual(len([r for r in primary if r['method'] in R.BASELINES]), 100)

    def test_missing_duplicate_or_foreign_seed_rejected(self):
        old, new = fixture()
        mutations = [new[:-1], new + [new[0]], [{**new[0], 'seed': 10}, *new[1:]]]
        for fresh in mutations:
            with self.assertRaises(ValueError):
                R.combine(old, fresh, 'path', endpoints=('toy',))

    def test_population_or_k_mismatch_rejected(self):
        old, new = fixture()
        for field, value in [('n_scored', 29), ('k', 4)]:
            fresh = [{**new[0], field: value}, *new[1:]]
            with self.assertRaises(ValueError):
                R.combine(old, fresh, 'path', endpoints=('toy',))

    def test_mixed_config_or_changed_shared_control_rejected(self):
        old, new = fixture()
        with self.assertRaises(ValueError):
            R.combine(old, [{**new[0], 'configuration': 'other'}, *new[1:]], 'path', endpoints=('toy',))
        old[5]['configuration'] = 'other'
        with self.assertRaises(ValueError):
            R.combine(old, new, 'path', endpoints=('toy',))

    def test_mean_sd_is_initialization_dispersion(self):
        old, new = fixture()
        primary, _ = R.combine(old, new, 'path', endpoints=('toy',))
        rows = R.summarize(primary)
        candidate = next(r for r in rows if r['method'] == 'synrfp_task_candidate')
        self.assertAlmostEqual(candidate['ami_mean'], .3045)
        self.assertGreater(candidate['ami_sd'], 0)
        self.assertFalse(candidate['independent_test_replicates'])
        self.assertEqual(candidate['n_cells'], 10)

    def test_nonfinite_metric_rejected(self):
        old, new = fixture()
        new[0]['ami'] = float('nan')
        with self.assertRaises(ValueError):
            R.combine(old, new, 'path', endpoints=('toy',))

    def test_authoritative_original_digest_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); native = root / 'native.json'; U.write(native, {'result': 1})
            table = root / 'clustering_cells.csv'
            R.write_csv(table, [dict(source_result_path=str(native), source_result_sha256=U.sha(native))])
            plan = root / 'report/plan.json'; U.write(plan, {'scope': 'original'})
            receipt = root / 'report/COMPLETE.json'
            U.write(receipt, dict(phase='complete', tasks={'clustering': {'complete': True, 'completed_cells': 1440}},
                                 plan_sha256=U.sha(plan), outputs_sha256={table.name: U.sha(table)}))
            self.assertEqual(len(R.original_rows(U.HashCache(), {}, root, root)), 1)
            U.write(native, {'result': 2})
            with self.assertRaises(ValueError):
                R.original_rows(U.HashCache(), {}, root, root)

    def test_shared_native_result_path_and_retained_centers_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            centers = root / 'centers.npy'; centers.write_bytes(b'original-centers')
            native = root / 'shared.json'
            U.write(native, dict(artifacts_sha256={str(centers): U.sha(centers)}))
            table = root / 'clustering_cells.csv'
            R.write_csv(table, [dict(source_result_path='', source_result_sha256='',
                                    result_path=str(native), result_sha256=U.sha(native))])
            plan = root / 'report/plan.json'; U.write(plan, {'scope': 'original'})
            U.write(root / 'report/COMPLETE.json', dict(phase='complete',
                    tasks={'clustering': {'complete': True, 'completed_cells': 1440}},
                    plan_sha256=U.sha(plan), outputs_sha256={table.name: U.sha(table)}))
            hashes = {}
            self.assertEqual(len(R.original_rows(U.HashCache(), hashes, root, root)), 1)
            self.assertIn(str(centers), hashes)
            centers.write_bytes(b'changed-centers!')
            with self.assertRaises(ValueError):
                R.original_rows(U.HashCache(), {}, root, root)


if __name__ == '__main__':
    unittest.main()
