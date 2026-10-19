"""Independent coverage, balancing, dependency and dispatch mutation controls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import clustering_parallel_v2 as P


def fixture():
    panels = {ds: {"shared_is_all": ds in {"schneider_ub", "uspto50k_ub", "mech_1", "mech_2"}}
              for ds in P.C.ENDPOINTS}
    tasks = P.C.expected_tasks(panels)
    costs = {(ds, seed): float((i+1)*10 + seed) for i, ds in enumerate(P.C.ENDPOINTS) for seed in range(10)}
    return tasks, costs


class PartitionTests(unittest.TestCase):
    def test_exact_240_native_cells_eight_disjoint_workers(self):
        tasks, costs = fixture()
        bins = P.partition(tasks, costs)
        self.assertEqual(len(bins), 8)
        self.assertTrue(P.validate_partition(tasks, bins))
        cells = [t for b in bins for g in b["groups"] for t in g["tasks"]]
        self.assertEqual({P.C.identity(t) for t in cells}, {P.C.identity(t) for t in tasks})
        self.assertEqual(len(cells), 240)
        self.assertEqual(sum(t["origin"] == "fresh" for t in cells), 200)
        for b in bins:
            for g in b["groups"]:
                self.assertEqual([t["population"] for t in g["tasks"]], P.C.POLICIES)

    def test_deterministic_and_balanced_under_input_permutation(self):
        tasks, costs = fixture()
        first = P.partition(tasks, costs)
        self.assertEqual(first, P.partition(list(reversed(tasks)), costs))
        loads = [b["cost"] for b in first]
        self.assertLessEqual(max(loads)-min(loads), max(costs.values()))

    def test_skewed_native_style_workload_improves_endpoint_modulo(self):
        tasks, costs = fixture()
        heavy = {"tpl_b", "tpl_ub", "syntemp_r0"}
        costs = {k: 1000 if k[0] in heavy else 10 for k in costs}
        balanced = max(b["cost"] for b in P.partition(tasks, costs))
        old = [sum(v for (ds, seed), v in costs.items() if P.C.ENDPOINTS.index(ds)%4 == w) for w in range(4)]
        self.assertLess(balanced, .3*max(old))

    def test_false_merge_split_changed_seed_and_origin_rejected(self):
        tasks, costs = fixture()
        original = P.partition(tasks, costs)
        for mutation in ("duplicate", "drop", "seed", "origin", "dependency"):
            bins = copy.deepcopy(original)
            pair = bins[0]["groups"][0]["tasks"]
            if mutation == "duplicate":
                bins[1]["groups"].append(copy.deepcopy(bins[0]["groups"][0]))
            elif mutation == "drop":
                bins[0]["groups"].pop()
            elif mutation == "seed":
                pair[0]["seed"] = 500
            elif mutation == "origin":
                pair[0]["origin"] = "population_identical_copy"
            else:
                pair.reverse()
            with self.assertRaises(ValueError):
                P.validate_partition(tasks, bins)

    def test_missing_bad_cost_and_unpaired_population_rejected(self):
        tasks, costs = fixture()
        for value in (0, -1, float("nan"), float("inf"), True):
            bad = dict(costs)
            bad[next(iter(bad))] = value
            with self.assertRaises(ValueError):
                P.partition(tasks, bad)
        with self.assertRaises(ValueError):
            P.partition(tasks[:-1], costs)

    def test_dispatch_calls_only_native_fitter_for_owned_cells(self):
        tasks, costs = fixture()
        bins = P.partition(tasks, costs)
        plan = {"assignments": bins, "native_plan_sha256": "native"}
        native = {"tasks": tasks}
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            native_root = base / "native"
            native_root.mkdir()
            (base/"plan.json").write_text("{}")
            for i in range(4):
                (native_root/f"worker{i}.lock").touch()
            outputs = {str(base/"proof"): "digest"}
            with patch.object(P, "ROOT", base), patch.object(P.C, "ROOT", native_root), \
                 patch.object(P, "verify", return_value=plan), patch.object(P, "admission_ready"), \
                 patch.object(P.C, "verify", return_value=native), patch.object(P.C, "fit_cell") as fit, \
                 patch.object(P, "all_native_receipts", return_value=(bins[3], outputs)):
                P.run(3)
                expected = [t for g in bins[3]["groups"] for t in g["tasks"]]
                self.assertEqual([call.args for call in fit.call_args_list], [(t, native) for t in expected])
                rec = json.loads((base/"WORKER_3_COMPLETE.json").read_text())
                self.assertEqual(rec["completed"], len(expected))
                self.assertEqual(rec["outputs_sha256"], outputs)

    def test_worker_completion_cannot_hide_missing_native_artifact(self):
        tasks, costs = fixture()
        plan = {"assignments": P.partition(tasks, costs)}
        with patch.object(P.C, "audit_cell", return_value=None):
            with self.assertRaisesRegex(ValueError, "Missing balanced"):
                P.all_native_receipts(0, plan, {"tasks": tasks})

    def test_pilot_must_release_cgroup_and_leave_memory_headroom(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"parallel"
            native = Path(folder)/"native"
            root.mkdir(); native.mkdir()
            (root/"PREFLIGHT_READY.json").write_text(json.dumps({"phase": "complete", "plan_sha256": "wrapper"}))
            pilot = {"phase": "ready", "plan_sha256": "native", "peak_rss_gib": 5.5,
                     "task": {"dataset": "tpl_b", "population": "all_rows", "seed": 0, "origin": "fresh"},
                     "model_and_metric_replay": True}
            path = native/"PILOT_READY.json"
            path.write_text(json.dumps(pilot))
            with patch.object(P,"ROOT",root), patch.object(P.C,"ROOT",native), patch.object(P.C,"sha",return_value="wrapper"), \
                 patch.object(P.subprocess,"run") as run:
                run.return_value.stdout = "active\n"
                with self.assertRaisesRegex(ValueError,"cgroup to release"):
                    P.admission_ready({"native_plan_sha256":"native"})
                run.return_value.stdout = "inactive\n"
                P.admission_ready({"native_plan_sha256":"native"})
                path.write_text(json.dumps({**pilot,"peak_rss_gib":9.2}))
                with self.assertRaisesRegex(ValueError,"headroom"):
                    P.admission_ready({"native_plan_sha256":"native"})
                path.write_text(json.dumps({**pilot,"model_and_metric_replay":False}))
                with self.assertRaisesRegex(ValueError,"replay proof"):
                    P.admission_ready({"native_plan_sha256":"native"})

    def test_finalizer_requires_all_worker_bindings_then_native_report_once(self):
        tasks,costs=fixture()
        bins=P.partition(tasks,costs)
        plan={"assignments":bins,"native_plan_sha256":"native"}
        outputs={"/native/proof.json":"proof"}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"parallel"; native=Path(folder)/"native"
            root.mkdir(); native.mkdir()
            for b in bins:
                count=sum(len(g["tasks"]) for g in b["groups"])
                (root/f"WORKER_{b['worker']}_COMPLETE.json").write_text(json.dumps({"phase":"complete","completed":count,
                    "expected":count,"plan_sha256":"wrapper","outputs_sha256":outputs}))
            def report():
                (native/"COMPLETE.json").write_text(json.dumps({"phase":"complete","completed":240}))
            with patch.object(P,"ROOT",root),patch.object(P.C,"ROOT",native),patch.object(P,"verify",return_value=plan), \
                 patch.object(P.C,"verify",return_value={"tasks":tasks}),patch.object(P.C,"sha",return_value="wrapper"), \
                 patch.object(P,"all_native_receipts",side_effect=lambda w,p,n:(bins[w],outputs)), \
                 patch.object(P.C,"audit_cell",return_value={"complete":True}) as audit, \
                 patch.object(P.C,"report",side_effect=report) as native_report, \
                 patch.object(P.C,"run",side_effect=AssertionError("Do not replay original worker dispatch")):
                P.finalize()
                self.assertEqual(audit.call_count,240)
                native_report.assert_called_once()
                self.assertEqual(json.loads((root/"COMPLETE.json").read_text())["completed"],240)
                bad=root/"WORKER_0_COMPLETE.json"
                value=json.loads(bad.read_text());value["completed"]-=1;bad.write_text(json.dumps(value))
                native_report.reset_mock()
                with self.assertRaisesRegex(ValueError,"completion binding"):
                    P.finalize()
                native_report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
