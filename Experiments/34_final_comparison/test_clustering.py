"""Mutation and independent native-oracle controls for final clustering."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from threadpoolctl import threadpool_limits

import clustering as C


class ContractTests(unittest.TestCase):
    def test_full_cell_universe_and_no_baseline_jobs(self):
        equal = {"schneider_ub", "uspto50k_ub", "mech_1", "mech_2"}
        panels = {ds: {"shared_is_all": ds in equal} for ds in C.ENDPOINTS}
        tasks = C.expected_tasks(panels)
        self.assertEqual(len(tasks), 240)
        self.assertEqual(len({C.identity(t) for t in tasks}), 240)
        self.assertEqual(sum(t["origin"] == "fresh" for t in tasks), 200)
        self.assertEqual(sum(t["origin"] == "population_identical_copy" for t in tasks), 40)
        self.assertFalse(any("fingerprint" in t for t in tasks))
        for ds in C.ENDPOINTS:
            self.assertEqual({t["seed"] for t in tasks if t["dataset"] == ds}, set(range(10)))

    def test_native_algorithm_recipe(self):
        self.assertEqual(C.NATIVE.SMALL_PARAMS, {"n_init": 10, "max_iter": 300, "tol": 0.0001,
                         "algorithm": "lloyd", "init": "k-means++"})
        self.assertEqual(C.NATIVE.LARGE_PARAMS["batch_size"], 8192)
        self.assertEqual(C.NATIVE.LARGE_PARAMS["n_init"], 3)
        self.assertEqual(C.NATIVE.LARGE_PARAMS["max_iter"], 100)
        self.assertEqual(C.NATIVE.LARGE_PARAMS["max_no_improvement"], 10)
        self.assertEqual(C.FP, "synrfp-wl-r2-b2048-parity-delta")

    def test_environment_mismatch_refused(self):
        expected = C.environment()
        expected["scikit-learn"] = "invalid-version"
        with self.assertRaisesRegex(ValueError, "native synrfp"):
            C.check_environment(expected)

    def test_source_mutation_refused_even_same_size(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "source"
            p.write_text("old")
            pins = {str(p): C.sha(p)}
            p.write_text("new")
            with self.assertRaisesRegex(ValueError, "Changed frozen"):
                C.check_pins(pins, fresh=True)

    def test_wrong_known_k_and_population_refused(self):
        panel = {"k": 2, "n_rows": 6, "params": C.NATIVE.SMALL_PARAMS}
        rec = {"dataset": "mech_2", "fingerprint": "drfp", "seed": 0, "k": 2, "n_rows": 6,
               "method": "kmeans", "params": C.NATIVE.SMALL_PARAMS, "input_policy": "n1", "row_policy": "all_rows"}
        C.validate_native_record(rec, "mech_2", "drfp", "all_rows", 0, panel)
        for key, value in (("k", 3), ("row_policy", "shared_valid_all_six"), ("seed", 1),
                           ("params", {**C.NATIVE.SMALL_PARAMS, "n_init": 1}), ("fingerprint", C.LEGACY)):
            bad = {**rec, key: value}
            with self.assertRaises(ValueError):
                C.validate_native_record(bad, "mech_2", "drfp", "all_rows", 0, panel)

    def test_score_mutation_and_false_merge_rejected(self):
        a = {"label": {"ami": 1.0, "ari": 0.9}}
        self.assertTrue(C.score_equal(a, copy.deepcopy(a)))
        self.assertFalse(C.score_equal(a, {"label": {"ami": 1.0, "ari": 0.5}}))
        self.assertFalse(C.score_equal(a, {"other": {"ami": 1.0, "ari": 0.9}}))

    def test_native_model_feature_and_dataset_binding_mutations_refused(self):
        panel = {"k": 2, "n_rows": 6, "params": C.NATIVE.SMALL_PARAMS, "source": "mech",
                 "baseline_feature_sha256": {"drfp": "feature-digest"}, "dataset_sha256": "dataset-digest"}
        rec = {"dataset": "mech_2", "fingerprint": "drfp", "seed": 0, "k": 2, "n_rows": 6,
               "method": "kmeans", "params": C.NATIVE.SMALL_PARAMS, "input_policy": "n1", "row_policy": "all_rows",
               "source_sha256": {"/native/features/mech/drfp.npz": "feature-digest",
                                 "/native/tables/ds_mech_2.pkl": "dataset-digest"}}
        C.validate_native_record(rec, "mech_2", "drfp", "all_rows", 0, panel)
        for key in rec["source_sha256"]:
            bad = copy.deepcopy(rec)
            bad["source_sha256"][key] = "mutated"
            with self.assertRaisesRegex(ValueError, "bindings differ"):
                C.validate_native_record(bad, "mech_2", "drfp", "all_rows", 0, panel)


class ActualFitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.ds = pd.DataFrame({"row": np.arange(6), "r_id": [f"r{i}" for i in range(6)],
                                "y": [0, 0, 0, 1, 1, 1], "label": ["a", "a", "a", "b", "b", "b"]})
        packed = np.zeros((6, 256), np.uint8)
        packed[:3, :8] = 255
        packed[3:, 8:16] = 255
        self.cache = self.root / "features.npz"
        np.savez(self.cache, X=packed, n_bits=np.int64(2048), ok=np.ones(6, bool))
        self.population = self.root / "population.npz"
        np.savez(self.population, valid=np.ones(6, bool), shared_valid=np.ones(6, bool), all_rows=np.ones(6, bool))
        self.panel = {"source": "fixture", "population_file": str(self.population), "params": C.NATIVE.SMALL_PARAMS,
                      "n_rows": 6, "k": 2, "label": "label", "method": "kmeans", "shared_is_all": True}
        self.plan = {"panels": {"mech_2": self.panel}, "sources": {"fixture": {"path": str(self.cache)}},
                     "pins": {str(self.cache): C.sha(self.cache)}}
        (self.root / "plan.json").write_text(json.dumps(self.plan))
        self.task = {"dataset": "mech_2", "population": "all_rows", "seed": 0, "origin": "fresh"}
        self.cfg = {"datasets": {"mech_2": {"label": "label"}}}
        self.patches = [patch.object(C, "ROOT", self.root), patch.object(C, "load_config", return_value=self.cfg),
                        patch.object(C, "dataset_table", return_value=self.ds)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.directory.cleanup()

    def test_real_native_fit_persistence_metric_and_center_replay(self):
        self.assertTrue(C.fit_cell(self.task, self.plan))
        rec = C.audit_cell(self.task, self.plan, replay=True)
        X = C.feature(self.cache).dense()
        with threadpool_limits(limits=2):
            native = KMeans(n_clusters=2, random_state=0, **C.NATIVE.SMALL_PARAMS).fit_predict(X)
        with np.load(rec["assignment_path"]) as z:
            np.testing.assert_array_equal(z["labels"], native)
        self.assertEqual(rec["scores"]["label"]["ami"], 1.0)
        self.assertFalse(C.fit_cell(self.task, self.plan))

    def test_identical_population_copy_has_zero_new_fit_and_bound_source(self):
        C.fit_cell(self.task, self.plan)
        copied = {**self.task, "population": "shared_valid_all_six", "origin": "population_identical_copy"}
        with patch.object(C, "KMeans", side_effect=AssertionError("copy must not refit")):
            C.fit_cell(copied, self.plan)
        rec = C.audit_cell(copied, self.plan, replay=True)
        self.assertEqual(rec["fit_seconds"], 0)
        self.assertEqual(rec["copied_result_sha256"], C.sha(C.stem(self.task).with_suffix(".json")))

    def test_real_native_minibatch_fit_matches_independent_native_oracle(self):
        name = "tpl_b"
        panel = {**self.panel, "params": C.NATIVE.LARGE_PARAMS, "method": "minibatch_kmeans"}
        plan = {**self.plan, "panels": {name: panel}}
        self.cfg["datasets"][name] = {"label": "label"}
        task = {**self.task, "dataset": name}
        C.fit_cell(task, plan)
        rec = C.audit_cell(task, plan, replay=True)
        with threadpool_limits(limits=2):
            native = MiniBatchKMeans(n_clusters=2, random_state=0, **C.NATIVE.LARGE_PARAMS).fit_predict(C.feature(self.cache).dense())
        with np.load(rec["assignment_path"]) as z:
            np.testing.assert_array_equal(z["labels"], native)

    def test_nonidentical_population_copy_refused_before_writing(self):
        C.fit_cell(self.task, self.plan)
        shared = np.array([True, True, True, True, True, False])
        np.savez(self.population, valid=np.ones(6, bool), shared_valid=shared, all_rows=np.ones(6, bool))
        copied = {**self.task, "population": "shared_valid_all_six", "origin": "population_identical_copy"}
        with self.assertRaisesRegex(ValueError, "nonidentical population"):
            C.fit_cell(copied, self.plan)
        self.assertFalse(C.stem(copied).with_suffix(".json").exists())

    def test_assignment_row_permutation_and_mask_mutations_rejected(self):
        C.fit_cell(self.task, self.plan)
        rec = C.audit_cell(self.task, self.plan)
        with np.load(rec["assignment_path"]) as z:
            values = {k: z[k] for k in z.files}
        for key in ("source_row", "r_id", "y", "valid", "used", "labels"):
            bad = {k: v.copy() for k, v in values.items()}
            if key in ("valid", "used"):
                bad[key][0] = False
            elif key == "labels":
                bad[key][0] = -1
            else:
                bad[key] = bad[key][::-1]
            out = self.root / "bad.npz"
            np.savez(out, **bad)
            with self.assertRaises(ValueError):
                C.validate_assignment(out, self.ds, np.ones(6, bool), np.ones(6, bool))

    def test_metric_mutation_fails_independent_assignment_replay(self):
        C.fit_cell(self.task, self.plan)
        rec = C.audit_cell(self.task, self.plan)
        rec["scores"]["label"]["ami"] = 0.25
        with self.assertRaisesRegex(ValueError, "metric replay"):
            C.verify_metric_replay(self.task, self.plan, rec)

    def test_changed_saved_centers_or_result_identity_refused(self):
        C.fit_cell(self.task, self.plan)
        rec = C.audit_cell(self.task, self.plan)
        with Path(rec["centers_path"]).open("ab") as handle:
            handle.write(b"mutation")
        with self.assertRaisesRegex(ValueError, "Changed final clustering artifact"):
            C.audit_cell(self.task, self.plan)


if __name__ == "__main__":
    unittest.main()
