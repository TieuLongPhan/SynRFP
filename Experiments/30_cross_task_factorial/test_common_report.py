"""Independent weighting, direction, completeness and source-binding controls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("common30_under_test", Path(__file__).with_name("common_report.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def cell(task, config, score, regime=None, source=None, endpoint=None, fold="0", seed="42"):
    regime = regime or ("grouped_cv" if task in ("classification", "reagent", "condition")
                        else "heuristic_rowcount_kmeans" if task == "clustering"
                        else "external_fixed_test" if task == "feasibility" else "original_regime")
    source = source or task
    return dict(task=task, config=config, score=float(score), regime=regime, source=source,
                endpoint=endpoint or source, family=source, fold=fold, seed=seed,
                encoding_coverage=1., test_rows=10, coverage_scope="test_fixture", origin="fixture")


def catalogue():
    rows = []
    for task in C.TASKS:
        for config in ("A", "B"):
            score = (1 if config == "A" else 2) if task == "barrier" else (2 if config == "A" else 1)
            rows.append(cell(task, config, score))
            if task in ("reagent", "condition"):
                rows.append(cell(task, config, 1 if config == "A" else 2, regime="native", fold="native"))
    return pd.DataFrame(rows)


class CommonReportContracts(unittest.TestCase):
    def test_lower_error_higher_other_metrics_and_separate_native_view(self):
        result = C.analyze(catalogue(), ["A", "B"])
        endpoints = result["endpoint_summary"]
        barriers = endpoints[endpoints.task.eq("barrier")].set_index("config")
        self.assertEqual(barriers.loc["A", "rank"], 1)
        self.assertEqual(barriers.loc["A", "normalized_regret"], 0)
        self.assertEqual(barriers.loc["B", "normalized_regret"], 1)
        main = result["common_candidates"].query("view == 'grouped_main'").set_index("config")
        native = result["common_candidates"].query("view == 'native_reference'").set_index("config")
        self.assertEqual(main.loc["A", "macro_rank"], 1)
        self.assertAlmostEqual(native.loc["A", "macro_rank"], 9/7)
        self.assertEqual(main.loc["A", "represented_tasks"], 7)
        self.assertIn("exploratory", main.loc["A", "interpretation"])
        self.assertNotIn("selected", result)

    def test_equal_task_weight_does_not_depend_on_dataset_or_fit_count(self):
        frame = catalogue()
        frame = frame[~frame.task.eq("classification")]
        for i in range(40):
            for config in ("A", "B"):
                frame = pd.concat([frame, pd.DataFrame([cell("classification", config, 1 if config == "A" else 2,
                                      source="schneider", endpoint="schneider_"+str(i))])], ignore_index=True)
        result = C.analyze(frame, ["A", "B"])["common_candidates"].query("view == 'grouped_main'").set_index("config")
        self.assertAlmostEqual(result.loc["A", "macro_rank"], 8/7)
        self.assertAlmostEqual(result.loc["B", "macro_rank"], 13/7)

    def test_condition_fold_means_precede_optimizer_and_partition_variation(self):
        frame = catalogue()
        frame = frame[~(frame.task.eq("condition") & frame.regime.eq("grouped_cv"))]
        rows = []
        for config in ("A", "B"):
            offset = 0 if config == "A" else -1
            # Deliberately unequal initialization counts independently detect
            # accidental weighting by number of models instead of fold means.
            for seed, score in enumerate([0, 1, 2, 3, 4]):
                rows.append(cell("condition", config, score+offset, fold="fold0", seed=str(seed)))
            rows.append(cell("condition", config, 10+offset, fold="fold1", seed="0"))
        result = C.analyze(pd.concat([frame,pd.DataFrame(rows)],ignore_index=True), ["A", "B"])
        endpoint = result["endpoint_summary"].query("task == 'condition' and regime == 'grouped_cv' and config == 'A'").iloc[0]
        self.assertEqual(endpoint.mean_score, 6.)
        self.assertEqual(endpoint.partitions, 2)
        self.assertAlmostEqual(endpoint.mean_within_seed_sd, np.std([0,1,2,3,4], ddof=1))
        self.assertAlmostEqual(endpoint.between_partition_sd, np.std([2,10], ddof=1))
        self.assertEqual(endpoint.completed_cells, 6)

    def test_exact_ties_have_zero_regret_and_average_rank(self):
        frame = catalogue()
        frame.loc[frame.task.eq("clustering"), "score"] = .5
        endpoint = C.analyze(frame, ["A", "B"])["endpoint_summary"].query("task == 'clustering'")
        self.assertTrue(endpoint["rank"].eq(1.5).all())
        self.assertTrue(endpoint.normalized_regret.eq(0).all())

    def test_missing_construction_duplicate_or_seed_population_is_refused(self):
        frame = catalogue()
        with self.assertRaises(ValueError): C.analyze(frame.iloc[1:], ["A", "B"])
        with self.assertRaises(ValueError): C.analyze(pd.concat([frame,frame.iloc[:1]],ignore_index=True), ["A", "B"])
        changed=frame.copy();changed.loc[0,"seed"]="999"
        with self.assertRaises(ValueError): C.analyze(changed, ["A", "B"])
        changed=frame.copy();changed.loc[0,"score"]=np.nan
        with self.assertRaises(ValueError): C.analyze(changed, ["A", "B"])

    def test_family_dependencies_and_native_group_aliases_are_explicit(self):
        self.assertEqual(C.family("barrier","gdb7"), C.family("barrier","rdb7"))
        self.assertEqual(C.family("barrier","e2"), C.family("barrier","sn2"))
        self.assertEqual(C.family("barrier","sn2"), C.family("barrier","e2sn2"))
        self.assertEqual(C.family("classification","syntemp_r0"),C.family("classification","syntemp_r2"))
        self.assertEqual(C.normalize_regime("native_published"), "native")
        self.assertEqual(C.normalize_regime("group_disjoint"), "grouped_cv")
        self.assertEqual(C.normalize_regime("group_cv"), "grouped_cv")
        self.assertEqual(C.normalize_regime("reactant_group_holdout"), "grouped_holdout")
        with self.assertRaises(ValueError): C.family("classification","ehreact")

    def test_csv_and_plan_mutation_or_partial_canonical_audit_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=root/"plan.json";csv=root/"cells.csv";receipt=root/"COMPLETE.json"
            C.S.write(plan,dict(expected_reused_models=0))
            pd.DataFrame([dict(phase="complete",score=1)]).to_csv(csv,index=False)
            value=dict(phase="complete",completed=1,expected=1,reused=0,
                       canonical_metrics_and_artifacts_audited=1,plan_sha256=C.S.sha(plan),
                       outputs_sha256={"cells.csv":C.S.sha(csv)})
            C.S.write(receipt,value)
            frame,identity=C.canonical_cells(csv,receipt,plan,1,new=True)
            self.assertEqual(len(frame),1);self.assertEqual(identity["complete_sha256"],C.S.sha(receipt))
            C.S.write(receipt,dict(value,canonical_metrics_and_artifacts_audited=0))
            with self.assertRaises(ValueError): C.canonical_cells(csv,receipt,plan,1,new=True)
            C.S.write(receipt,value);csv.write_text("changed table\n")
            with self.assertRaises(ValueError): C.canonical_cells(csv,receipt,plan,1,new=True)
            pd.DataFrame([dict(phase="complete",score=1)]).to_csv(csv,index=False)
            plan.write_text("changed plan")
            with self.assertRaises(ValueError): C.canonical_cells(csv,receipt,plan,1,new=True)


if __name__ == "__main__":
    unittest.main()
