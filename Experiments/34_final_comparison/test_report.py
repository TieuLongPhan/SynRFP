"""Scientific identity and aggregation regressions for the final report."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("final_report", Path(__file__).with_name("report.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def preset():
    return dict(configuration_selected=R.SELECTED,
                configuration=dict(tokenizer="wl", radius=2, sketch="parity", mode="delta", n_bits=2048, seed=1),
                interpretation="exploratory_after_prior_test_results_viewed", test_sets_independent=False)


def panel():
    return [dict(dataset="a", seed=s, method=m) for s in range(5) for m in R.METHODS]


class IdentityTests(unittest.TestCase):
    def test_exact_preset_accepts(self):
        R.validate_preset(preset())

    def test_historical_width_refused(self):
        value = preset()
        value["configuration"]["n_bits"] = 1024
        with self.assertRaisesRegex(ValueError, "width/sign"):
            R.validate_preset(value)

    def test_unsigned_or_union_replacement_refused(self):
        for wrong in ("union", "roles", "delta_union"):
            value = preset()
            value["configuration"]["mode"] = wrong
            with self.assertRaises(ValueError):
                R.validate_preset(value)

    def test_independent_selection_claim_refused(self):
        value = preset()
        value["test_sets_independent"] = True
        with self.assertRaisesRegex(ValueError, "independent"):
            R.validate_preset(value)

    def test_missing_primary_is_partial_only(self):
        rows = [r for r in panel() if not (r["method"] == R.PRIMARY and r["seed"] == 4)]
        R.check_panel(rows, ("dataset", "seed"), 30, complete=False)
        with self.assertRaisesRegex(ValueError, "missing"):
            R.check_panel(rows, ("dataset", "seed"), 30)

    def test_same_count_wrong_split_seed_refused(self):
        rows = panel()
        rows[0]["seed"] = 91
        with self.assertRaises(ValueError):
            R.check_panel(rows, ("dataset", "seed"), 30)

    def test_duplicate_cell_refused(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            R.check_panel(panel() + [panel()[0]], ("dataset", "seed"), 30)

    def test_supervised_fingerprint_refused(self):
        rows = panel()
        rows[0]["method"] = "rxnfp"
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            R.check_panel(rows, ("dataset", "seed"), 30)

    def test_component_completion_needs_actual_output_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "cells.csv").write_text("x\n1\n")
            receipt = dict(outputs_sha256={"cells.csv": R.sha(path / "cells.csv")})
            R.receipt_outputs(receipt, path)
            (path / "cells.csv").write_text("x\n2\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                R.receipt_outputs(receipt, path)
            with self.assertRaisesRegex(ValueError, "catalog"):
                R.receipt_outputs({}, path)

    def test_metric_mutation_refused(self):
        with self.assertRaisesRegex(ValueError, "Canonical metric"):
            R.metric_equal(dict(mcc=.1), dict(mcc=.2))


class AggregationTests(unittest.TestCase):
    def test_classification_sd_is_across_repeat_means(self):
        rows = [dict(task="classification", dataset="a", method=R.PRIMARY,
                     model="mlp", protocol="grouped_5_repeats_5_folds", seed=r * 5 + f,
                     mcc=r + f * .1) for r in range(5) for f in range(5)]
        result = R.summarize(rows)[0]
        self.assertEqual(result["n_cells"], 25)
        self.assertEqual(result["n_summary_units"], 5)
        self.assertAlmostEqual(result["mcc_mean"], 2.2)
        self.assertAlmostEqual(result["mcc_sd"], 1.5811388300841898)
        self.assertFalse(result["independent_test_replicates"])

    def test_condition_optimizer_variation_is_not_fold_variation(self):
        rows = [dict(task="condition", dataset="a", method=R.PRIMARY, protocol="group_disjoint",
                     fold=f, seed=s, split=f"group_fold{f}_seed{s}", micro_f1=f + s * .1)
                for f in range(5) for s in range(5)]
        result = R.summarize(rows)[0]
        self.assertEqual(result["n_cells"], 25)
        self.assertEqual(result["n_summary_units"], 5)
        self.assertAlmostEqual(result["micro_f1_sd"], 1.5811388300841898)
        self.assertNotEqual(result["micro_f1_sd"], result["micro_f1_raw_cell_sd"])

    def test_incomplete_condition_fold_seed_population_refused(self):
        rows = [dict(task="condition", dataset="a", method=R.PRIMARY,
                     protocol="group_disjoint", fold=0, seed=s, micro_f1=.3) for s in range(5)]
        with self.assertRaisesRegex(ValueError, "fold/seed"):
            R.summarize(rows)

    def test_ncs_iterations_are_ten_units_not_forty_independent_tests(self):
        rows = [dict(task="yield", dataset="ncs_bh", method=R.PRIMARY,
                     protocol="ncs_cluster_ood", split=f"i{r}_f{f}", r2=r * .01 + f * .001)
                for r in range(10) for f in range(4)]
        result = R.summarize(rows)[0]
        self.assertEqual(result["n_cells"], 40)
        self.assertEqual(result["n_summary_units"], 10)
        self.assertEqual(result["sd_scope"], "across_ten_iteration_means")

    def test_native_and_grouped_scopes_do_not_mix(self):
        rows = [dict(task="reagent", dataset="a", method=R.PRIMARY,
                     protocol=p, seed=s, split=str(s), micro_f1=s * .01)
                for p in ("native", "group_disjoint") for s in range(5)]
        result = R.summarize(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual({r["n_cells"] for r in result}, {5})

    def test_unrelated_task_metrics_never_average(self):
        rows = [dict(task="classification", dataset="a", method=R.PRIMARY, seed=0, mcc=.7),
                dict(task="yield", dataset="a", method=R.PRIMARY, split="random_1", r2=.3)]
        result = R.summarize(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(sum("mcc_mean" in r for r in result), 1)
        self.assertEqual(sum("r2_mean" in r for r in result), 1)


if __name__ == "__main__":
    unittest.main()
