"""Independent alignment and information-boundary regressions for evidence31."""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("ncs_evidence_analysis_tested", Path(__file__).with_name("analysis.py"))
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def grid():
    return {f"{t}-{r}-{s}-{m}": dict(tokenizer=t, radius=r, sketch=s, mode=m, n_bits=2048)
            for t, r, s, m in itertools.product(("wl", "morgan", "path"), (2, 3),
                                               ("parity", "minhash", "cw"),
                                               ("delta", "union", "delta_union", "roles"))}


def population(task="classification", endpoints=(("schneider_b", "schneider", 2.),), seeds=("42",), partitions=("0", "1")):
    rows = []
    for endpoint, family, effect in endpoints:
        for partition, seed, (name, settings) in itertools.product(partitions, seeds, grid().items()):
            # Partition baselines differ greatly; subtraction must pair within
            # each partition rather than zip rows in input or config order.
            base = 100 * int(partition) + 3 * settings["radius"] + int(seed)
            contribution = {"delta": 0., "union": -effect, "delta_union": effect, "roles": effect + 0.5}
            rows.append(dict(task=task, source=endpoint, family=family, endpoint=endpoint,
                             regime="grouped_cv", partition=partition, repeat_kind="heldout_group_fold",
                             seed=seed, config=name, score=base + contribution[settings["mode"]]))
    return pd.DataFrame(rows)


class PairingTests(unittest.TestCase):
    def test_shuffled_partition_pairing_has_exact_expected_effect(self):
        data = population().sample(frac=1, random_state=391).reset_index(drop=True)
        result = A.analyze(data, grid())["matched_mode_contrasts"]
        expected = dict(delta_union_minus_delta=2., roles_minus_delta=2.5,
                        roles_minus_delta_union=0.5, union_minus_delta=-2.)
        self.assertEqual(len(result), 2 * 18 * 4)
        for contrast, value in expected.items():
            np.testing.assert_allclose(result[result.contrast.eq(contrast)].difference, value)

    def test_one_missing_mode_cell_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "Incomplete paired"):
            A.analyze(population().iloc[1:], grid())

    def test_duplicate_cell_is_rejected(self):
        data = population()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            A.analyze(pd.concat([data, data.iloc[:1]], ignore_index=True), grid())

    def test_missing_optimizer_seed_for_one_config_is_rejected(self):
        data = population(task="condition", seeds=("42", "43", "44", "45", "46"))
        with self.assertRaisesRegex(ValueError, "seed populations"):
            A.analyze(data.iloc[1:], grid())

    def test_condition_initializations_are_averaged_inside_fold(self):
        data = population(task="condition", seeds=("42", "43", "44", "45", "46"))
        result = A.analyze(data, grid())
        matched = result["matched_mode_contrasts"]
        self.assertEqual(len(matched), 2 * 18 * 4)
        self.assertTrue(matched.left_raw_cells.eq(5).all())
        self.assertTrue(matched.right_raw_cells.eq(5).all())
        self.assertTrue(result["endpoint_contrasts"].paired_partitions.eq(2).all())

    def test_family_balance_does_not_weight_hierarchical_views_as_sources(self):
        data = population(endpoints=(("mech_1", "mech", 1.), ("mech_2", "mech", 3.),
                                    ("schneider_b", "schneider", 8.)))
        tasks = A.analyze(data, grid())["task_contrasts"]
        contrast = tasks[tasks.contrast.eq("delta_union_minus_delta")]
        # Mean(mech endpoints)=2, Schneider=8; balanced family mean=5.
        np.testing.assert_allclose(contrast.mean_difference, 5.)

    def test_barrier_benefit_has_opposite_sign_from_raw_rmse_difference(self):
        result = A.analyze(population(task="barrier"), grid())["matched_mode_contrasts"]
        np.testing.assert_allclose(result.benefit, -result.difference)

    def test_native_and_grouped_condition_views_do_not_mix(self):
        grouped = population(task="condition")
        native = grouped.copy()
        native["regime"] = "native"
        names = set(k for k, v in grid().items() if v["mode"] == "delta_union")
        native.loc[native.config.isin(names), "score"] += 10
        result = A.analyze(pd.concat([grouped, native], ignore_index=True), grid())["task_contrasts"]
        selected = result[result.contrast.eq("delta_union_minus_delta")]
        np.testing.assert_allclose(selected[selected.view.eq("grouped_main")].mean_difference, 2.)
        np.testing.assert_allclose(selected[selected.view.eq("native_reference")].mean_difference, 12.)

    def test_nonfinite_metric_is_rejected(self):
        data = population()
        data.loc[0, "score"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            A.analyze(data, grid())

    def test_cluster_initializations_are_never_called_folds(self):
        raw = pd.DataFrame(dict(dataset=["mech_2"] * 2, config=["example"] * 2,
                                ami=[0.3, 0.4], seed=[0, 1]))
        result = A.normalize_old(raw, "clustering")
        self.assertEqual(set(result.partition), {"cluster_init_0", "cluster_init_1"})
        self.assertEqual(set(result.repeat_kind), {"cluster_initialization"})

    def test_fixed_frea_test_seeds_share_one_test_population(self):
        raw = pd.DataFrame([dict(task="feasibility", source="feasibility", split=f"seed{i}",
                                 protocol="external_fixed_test", config="example", auroc=.8,
                                 seed=i, target="feasible") for i in range(5)])
        result = A.normalize_new(pd.concat([raw,
            pd.DataFrame([dict(task=t, source=t, split="native_seed42", protocol="native",
                               config="example", micro_f1=.7, rmse=1., seed=42, target="target")
                          for t in ("barrier", "reagent", "condition")])], ignore_index=True))
        fixed = result[result.task.eq("feasibility")]
        self.assertEqual(fixed.partition.nunique(), 1)


class CanonicalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.plan = self.path / "plan.json"
        A.write(self.plan, {"example": "fixed native plan"})
        self.table = self.path / "cells.csv"
        pd.DataFrame({"phase": ["complete"], "score": [.7]}).to_csv(self.table, index=False)
        self.other = self.path / "report.md"
        self.other.write_text("Bound report output.\n")
        self.receipt = self.path / "COMPLETE.json"
        A.write(self.receipt, dict(phase="complete", task="yield", cells=1,
            canonical_metric_and_artifact_checks_passed=True, configuration_selected=None,
            plan_sha256=A.sha(self.plan), outputs_sha256={"cells.csv": A.sha(self.table), "report.md": A.sha(self.other)}))

    def tearDown(self):
        self.tmp.cleanup()

    def admit(self):
        return A.verified_cells(self.table, self.receipt, self.plan, 1, "yield")

    def test_admits_only_bound_completed_csv(self):
        frame, source = self.admit()
        self.assertEqual(len(frame), 1)
        self.assertEqual(source["csv_sha256"], A.sha(self.table))

    def test_csv_mutation_is_rejected(self):
        self.table.write_text(self.table.read_text().replace("0.7", "0.9"))
        with self.assertRaisesRegex(ValueError, "Changed or unbound"):
            self.admit()

    def test_non_csv_bound_output_mutation_is_rejected(self):
        self.other.write_text("Mutated report.\n")
        with self.assertRaisesRegex(ValueError, "Changed or unbound"):
            self.admit()

    def test_plan_mutation_is_rejected(self):
        A.write(self.plan, {"example": "changed native plan"})
        with self.assertRaisesRegex(ValueError, "changed factorial"):
            self.admit()

    def test_incomplete_count_is_rejected(self):
        receipt = json.loads(self.receipt.read_text())
        receipt["cells"] = 0
        A.write(self.receipt, receipt)
        with self.assertRaisesRegex(ValueError, "completion ledger"):
            self.admit()

    def test_new_receipt_may_live_separately_from_exported_outputs(self):
        receipt = self.path / "local-root/COMPLETE.json"
        A.write(receipt, dict(phase="complete", completed=1, expected=1,
            canonical_metrics_and_artifacts_audited=1, plan_sha256=A.sha(self.plan),
            outputs_sha256={"cells.csv": A.sha(self.table), "report.md": A.sha(self.other)}))
        frame, _ = A.verified_cells(self.table, receipt, self.plan, 1)
        self.assertEqual(len(frame), 1)


class ProducerRoutingTests(unittest.TestCase):
    def test_attempt_receipt_binds_exact_status_and_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            A.write(root / "export/status.json", {"phase": "waiting_for_all_seven"})
            with patch.dict(A.os.environ, {"NCS_EVIDENCE_STAGE_ID": "a" * 32,
                                          "NCS_EVIDENCE_INPUT_KEY": "input-generation-7"}):
                A.producer_receipt(root, root / "export")
            receipt = A.read(root / "stages" / ("a" * 32) / "PRODUCER_READY.json")
            self.assertEqual(receipt["result_sha256"], A.sha(root / "export/status.json"))
            self.assertEqual(receipt["producer_sha256"], A.sha(A.__file__))
            self.assertEqual(receipt["input_key"], "input-generation-7")

    def test_unsafe_stage_path_is_rejected(self):
        with patch.dict(A.os.environ, {"NCS_EVIDENCE_STAGE_ID": "../different-run",
                                      "NCS_EVIDENCE_INPUT_KEY": "some-key"}):
            with self.assertRaisesRegex(ValueError, "routing identity"):
                A.producer_receipt()


if __name__ == "__main__":
    unittest.main()
