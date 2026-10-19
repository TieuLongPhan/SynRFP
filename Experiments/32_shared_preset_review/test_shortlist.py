"""Focused admission, deterministic ordering and completion mutation controls."""
from copy import deepcopy
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shared_preset_review", HERE / "shortlist.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def grid():
    values = itertools.product(("wl", "morgan", "path"), (2, 3),
              ("parity", "minhash", "cw"), ("delta", "union", "delta_union", "roles"))
    return {f"c{i:02d}": dict(zip(("tokenizer", "radius", "sketch", "mode"), v))
            for i, v in enumerate(values)}


def frames(configs):
    candidates, tasks, endpoints, families = [], [], [], []
    for view in sorted(S.VIEWS):
        for i, name in enumerate(configs):
            rank = float(i + 1) if view == "grouped_main" else float(72 - i)
            row = dict(config=name, view=view, macro_rank=rank, macro_regret=i / 72,
                       worst_task_rank=rank, encoding_coverage=1., represented_tasks=7)
            for task in S.TASKS:
                row[task + "_rank"] = rank
                row[task + "_regret"] = i / 72
                tasks.append(dict(config=name, task=task, view=view, rank=rank,
                    regret=i / 72, encoding_coverage=1., source_families=1))
                if view == "grouped_main":
                    endpoints.append(dict(config=name, task=task, endpoint=task,
                                          mean_score=float(i), metric="native"))
                    families.append(dict(config=name, task=task, family=task,
                                         mean_rank=rank, mean_regret=i / 72))
            candidates.append(row)
    return tuple(map(pd.DataFrame, (candidates, tasks, endpoints, families)))


def fixture(base):
    configs = grid()
    source_code = base / "common_source.py"
    source_code.write_text("# frozen independent common producer fixture\n")
    plan = base / "plan.json"
    S.write(plan, dict(configs=configs))
    sources = []
    for count, task in ((5040, "classification"), (4320, "clustering"),
                        (4608, "yield"), (6480, "new")):
        directory = base / task
        directory.mkdir()
        table = directory / "cells.csv"
        table.write_text("canonical fixture bytes\n")
        ledger = dict(phase="complete", configuration_selected=None, plan_sha256=S.sha(plan),
                      outputs_sha256={"cells.csv": S.sha(table)})
        if task == "new":
            ledger.update(completed=6480, expected=6480, canonical_metrics_and_artifacts_audited=6480,
                          reused=270)
        else:
            ledger.update(task=task, cells=count, canonical_metric_and_artifact_checks_passed=True)
        complete = directory / "COMPLETE.json"
        S.write(complete, ledger)
        sources.append(dict(csv=str(table), csv_sha256=S.sha(table), complete=str(complete),
                       complete_sha256=S.sha(complete), plan=str(plan), plan_sha256=S.sha(plan), cells=count))
    output = base / "common_outputs"
    output.mkdir()
    for name, frame in zip(("common_candidates", "task_view_summary", "endpoint_summary", "family_summary"), frames(configs)):
        frame.to_csv(output / (name + ".csv"), index=False)
    receipt = dict(phase="complete", configuration_selected=None, logical_input_cells=20448,
        configurations=72, tasks=list(S.TASKS), views=sorted(S.VIEWS), test_sets_independent=False,
        source_sha256=S.sha(source_code), output_root=str(output), sources=sources,
        outputs_sha256={p.name: S.sha(p) for p in output.glob("*.csv")})
    path = base / "COMMON_TASK_ANALYSIS_COMPLETE.json"
    S.write(output / "COMPLETE.json", receipt)
    S.write(path, receipt)
    return path, source_code


class ShortlistTests(unittest.TestCase):
    def test_global_order_and_native_sensitivity_are_distinct(self):
        configs = grid()
        selected, output = S.analyze(*frames(configs), configs)
        self.assertEqual(selected, ["c00", "c01", "c02"])
        first = output["top3_candidates"].query("view == 'grouped_main'").iloc[0]
        self.assertEqual(int(first.native_reference_position), 72)
        self.assertEqual(len(output["top3_task_diagnostics"]), 42)

    def test_ties_use_regret_then_worst_task_then_lexical(self):
        configs = grid()
        values = list(frames(configs))
        candidates = values[0]
        mask = candidates.view.eq("grouped_main") & candidates.config.isin(["c00", "c01", "c02", "c03"])
        candidates.loc[mask, ["macro_rank", "macro_regret", "worst_task_rank"]] = [1., .1, 4.]
        candidates.loc[mask & candidates.config.eq("c00"), "macro_regret"] = .2
        candidates.loc[mask & candidates.config.eq("c03"), "worst_task_rank"] = 3.
        selected, _ = S.analyze(*values, configs)
        self.assertEqual(selected, ["c03", "c01", "c02"])
        shuffled = [value.sample(frac=1, random_state=12) for value in values]
        self.assertEqual(S.analyze(*shuffled, configs)[0], selected)

    def test_reject_missing_configuration(self):
        configs = grid()
        values = list(frames(configs))
        values[0] = values[0][~values[0].config.eq("c71")]
        with self.assertRaisesRegex(ValueError, "Missing configuration"):
            S.analyze(*values, configs)

    def test_reject_missing_task(self):
        configs = grid()
        values = list(frames(configs))
        values[1] = values[1][~values[1].task.eq("barrier")]
        with self.assertRaisesRegex(ValueError, "task coverage"):
            S.analyze(*values, configs)

    def test_waits_without_reading_partial_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            result = S.once(base / "missing", base / "local", base / "export")
            self.assertEqual(result["phase"], "waiting_for_all_seven")
            self.assertEqual(result["refits_started"], 0)
            self.assertFalse((base / "local/COMPLETE.json").exists())

    def test_verify_complete_and_reject_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = fixture(Path(temp))
            self.assertEqual(len(S.verify_common(path, source)[2]), 72)
            receipt = S.read(path)
            table = Path(receipt["sources"][0]["csv"])
            table.write_text("changed bytes\n")
            with self.assertRaisesRegex(ValueError, "Changed or unbound"):
                S.verify_common(path, source)

    def test_reject_missing_source_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = fixture(Path(temp))
            receipt = S.read(path)
            del receipt["sources"][0]["csv_sha256"]
            S.write(path, receipt)
            S.write(Path(receipt["output_root"]) / "COMPLETE.json", receipt)
            with self.assertRaisesRegex(ValueError, "source field"):
                S.verify_common(path, source)

    def test_reject_false_completion_or_selected_preset(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = fixture(Path(temp))
            original = S.read(path)
            for field, value in (("logical_input_cells", 20447), ("configurations", 71),
                                 ("configuration_selected", "c00")):
                receipt = deepcopy(original)
                receipt[field] = value
                S.write(path, receipt)
                with self.assertRaisesRegex(ValueError, "complete seven-task"):
                    S.verify_common(path, source)

    def test_output_mutation_and_completed_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            path, source = fixture(base)
            result = S.once(path, base / "local", base / "export", source)
            self.assertEqual(result["shortlisted_configurations"], ["c00", "c01", "c02"])
            self.assertIsNone(result["configuration_selected"])
            self.assertEqual(result, S.once(path, base / "local", base / "export", source))
            (base / "export/top3_candidates.csv").write_text("modified output\n")
            with self.assertRaisesRegex(ValueError, "Changed or unbound"):
                S.once(path, base / "local", base / "export", source)


if __name__ == "__main__":
    unittest.main()
