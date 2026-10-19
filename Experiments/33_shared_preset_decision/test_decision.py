"""Decision boundaries: no partial selection, no task winners, no invented reuse."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

spec = importlib.util.spec_from_file_location("decision33_tested", Path(__file__).with_name("decision.py"))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def fixture():
    configs = {"c" + str(i).zfill(2): {"radius": 2, "bits": 2048} for i in range(72)}
    ranking = pd.DataFrame([
        dict(config=name, view=view, macro_rank=i + 1., macro_regret=i / 72.,
             worst_task_rank=70., represented_tasks=7)
        for view in ("grouped_main", "native_reference") for i, name in enumerate(configs)
    ])
    diagnostics = pd.DataFrame([
        dict(config=name, view=view, task=task, rank=70. if task == "yield" else 1.,
             regret=.8 if task == "yield" else .01, encoding_coverage=1., source_families=2)
        for name in list(configs)[:3] for view in ("grouped_main", "native_reference") for task in D.R.TASKS
    ])
    receipt = dict(phase="complete", configuration_selected=None, configurations=72,
        logical_input_cells=20448, tasks=list(D.R.TASKS), primary_view="grouped_main",
        order=D.R.ORDER, test_sets_independent=False, shortlisted_configurations=list(configs)[:3])
    return receipt, configs, list(configs)[:3], dict(all_72_rankings=ranking, top3_task_diagnostics=diagnostics)


class DecisionTests(unittest.TestCase):
    def test_global_compromise_keeps_weak_task_and_two_sensitivity_candidates(self):
        result = D.choose(*fixture())
        self.assertEqual(result["configuration_selected"], "c00")
        self.assertEqual(result["sensitivity_configurations"], ["c01", "c02"])
        self.assertTrue(any(d["task"] == "yield" and d["rank"] == 70 for d in result["task_diagnostics"]))
        self.assertEqual(len(result["task_diagnostics"]), 14)

    def test_fixed_lexical_tiebreak_does_not_follow_input_row_order(self):
        receipt, configs, selected, tables = fixture()
        frame = tables["all_72_rankings"]
        frame.loc[frame.config.isin(selected), ["macro_rank", "macro_regret", "worst_task_rank"]] = 1.
        tables["all_72_rankings"] = frame.iloc[::-1].copy()
        self.assertEqual(D.choose(receipt, configs, selected, tables)["configuration_selected"], "c00")

    def test_partial_count_or_missing_configuration_is_refused(self):
        for field, value in (("logical_input_cells", 20447), ("configurations", 71), ("phase", "running")):
            data = list(fixture())
            data[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                D.choose(*data)
        data = list(fixture())
        data[1].pop("c71")
        with self.assertRaises(ValueError):
            D.choose(*data)

    def test_changed_shortlist_or_prior_source_selection_is_refused(self):
        for key, value in (("shortlisted_configurations", ["c01", "c00", "c02"]),
                           ("configuration_selected", "c00"), ("test_sets_independent", True)):
            data = list(fixture())
            data[0][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                D.choose(*data)

    def test_missing_task_diagnostics_is_refused(self):
        data = list(fixture())
        data[3]["top3_task_diagnostics"] = data[3]["top3_task_diagnostics"].iloc[1:]
        with self.assertRaises(ValueError):
            D.choose(*data)

    def test_waiting_does_not_invoke_or_mutate_review_and_records_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(D.R, "ROOT", root / "absent32"), patch.object(D.R, "once") as review:
                status = D.once(root / "state33", root / "output33")
            review.assert_not_called()
            self.assertIsNone(status["configuration_selected"])
            self.assertEqual(status["authority"], D.AUTHORITY)
            self.assertEqual(status["phase"], "waiting_for_verified_shortlist")
            self.assertEqual(status["new_fits_launched"], 0)
            self.assertFalse((root / "state33/FROZEN_PRESET.json").exists())

    def test_followup_counts_are_inventory_and_never_invent_reuse_or_completion(self):
        report = D.followup("c00")
        self.assertEqual(report["primary_preset_protocol_cells"], 1373)
        self.assertEqual(report["factorial_cells_per_configuration"], 284)
        self.assertFalse(report["all_benchmarks_complete"])
        self.assertEqual(report["new_fits_launched"], 0)
        self.assertEqual(set(row["task"] for row in report["entries"]), set(D.R.TASKS))
        for row in report["entries"]:
            self.assertIsNone(row["missing_cells"])
            self.assertEqual(row["reusable_cells_verified"], 0)

    def test_cli_refuses_concurrent_owned_controller_without_unlinking_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "controller.lock"
            with path.open("a+") as handle:
                inode = path.stat().st_ino
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(D, "ROOT", root), patch("sys.argv", ["decision.py", "--once"]), \
                        patch.object(D, "once") as run, self.assertRaises(BlockingIOError):
                    D.main()
                run.assert_not_called()
                self.assertEqual(path.stat().st_ino, inode)

    def completed_fixture(self, temporary):
        root, output = Path(temporary) / "state", Path(temporary) / "export"
        root.mkdir(); output.mkdir()
        incoming = Path(temporary) / "input.json"
        D.write(incoming, {"complete": True})
        D.write(output / "FROZEN_PRESET.json", {"configuration_selected": "c00"})
        D.write(root / "FROZEN_PRESET.json", {"configuration_selected": "c00"})
        runtime = D.bindings()
        result = dict(phase="complete", runtime_bindings=runtime, authority=D.AUTHORITY,
            output_root=str(output.resolve()), input_sha256={str(incoming): D.sha(incoming)},
            outputs_sha256={"FROZEN_PRESET.json": D.sha(output / "FROZEN_PRESET.json")})
        D.write(output / "COMPLETE.json", result)
        return root, output, incoming, runtime, result

    def test_completed_input_mutation_is_refused_before_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, incoming, runtime, result = self.completed_fixture(temporary)
            D.write(incoming, {"complete": False})
            with patch.object(D.R, "once") as review, self.assertRaises(ValueError):
                D.verify_completed(result, root, output, runtime)
            review.assert_not_called()

    def test_completed_output_or_local_copy_or_producer_mutation_is_refused(self):
        for mutation in ("output", "local_copy", "producer"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root, output, incoming, runtime, result = self.completed_fixture(temporary)
                with patch.object(D.R, "once"):
                    self.assertEqual(D.verify_completed(result, root, output, runtime), result)
                    if mutation == "output":
                        D.write(output / "FROZEN_PRESET.json", {"configuration_selected": "c01"})
                    elif mutation == "local_copy":
                        D.write(root / "FROZEN_PRESET.json", {"configuration_selected": "c01"})
                    else:
                        runtime = copy.deepcopy(runtime)
                        runtime[next(iter(runtime))] = "0" * 64
                    with self.assertRaises(ValueError):
                        D.verify_completed(result, root, output, runtime)


if __name__ == "__main__":
    unittest.main()
