"""A fixed, exploratory global shortlist from completed seven-task factorials.

This sidecar reads reports only. It never fits models or chooses a final preset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COMMON_SOURCE = HERE.parent / "30_cross_task_factorial/common_report.py"
COMMON = Path.home() / ".local/share/synrfp-runs/cross_task_factorial_v1/COMMON_TASK_ANALYSIS_COMPLETE.json"
ROOT = Path.home() / ".local/share/synrfp-runs/shared_preset_review_v1"
OUTPUT = HERE / "publication_metadata/shared_preset_review_v1"
TASKS = ("classification", "clustering", "yield", "barrier", "feasibility", "reagent", "condition")
VIEWS = {"grouped_main", "native_reference"}
ORDER = ["macro_rank", "macro_regret", "worst_task_rank", "config"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check(path, digest):
    if not digest or sha(path) != digest:
        raise ValueError("Changed or unbound input: " + str(path))


def bound_outputs(directory, outputs):
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("Missing bound report outputs")
    for name, digest in outputs.items():
        if Path(name).name != name:
            raise ValueError("Unexpected report output path")
        check(Path(directory) / name, digest)


def verify_common(path, common_source=COMMON_SOURCE):
    """Verify the whole completed common ledger and all four source ledgers."""
    path = Path(path)
    receipt = read(path)
    if (receipt.get("phase") != "complete" or receipt.get("configuration_selected") is not None
            or receipt.get("logical_input_cells") != 20448 or receipt.get("configurations") != 72
            or set(receipt.get("tasks", [])) != set(TASKS) or set(receipt.get("views", [])) != VIEWS
            or receipt.get("test_sets_independent") is not False):
        raise ValueError("The complete seven-task exploratory ledger is required")
    check(common_source, receipt.get("source_sha256"))
    directory = Path(receipt["output_root"])
    if read(directory / "COMPLETE.json") != receipt:
        raise ValueError("Common receipt and output completion ledger differ")
    bound_outputs(directory, receipt.get("outputs_sha256"))
    sources = receipt.get("sources", [])
    if len(sources) != 4 or sorted(s.get("cells", -1) for s in sources) != [4320, 4608, 5040, 6480]:
        raise ValueError("Missing canonical factorial source")
    plans, identities = {}, []
    for source in sources:
        for field in ("csv", "complete", "plan"):
            if field not in source or field + "_sha256" not in source:
                raise ValueError("Missing bound canonical source field")
            check(source[field], source[field + "_sha256"])
        ledger = read(source["complete"])
        if (ledger.get("phase") != "complete" or ledger.get("configuration_selected") is not None
                or ledger.get("plan_sha256") != source["plan_sha256"]):
            raise ValueError("Changed canonical source ledger")
        count = source["cells"]
        if count == 6480:
            if (ledger.get("completed") != 6480 or ledger.get("expected") != 6480
                    or ledger.get("canonical_metrics_and_artifacts_audited") != 6480
                    or ledger.get("reused") != 270):
                raise ValueError("Incomplete four-task canonical audit")
            identity = "barrier_feasibility_reagent_condition"
        else:
            expected = {5040: "classification", 4320: "clustering", 4608: "yield"}[count]
            if (ledger.get("task") != expected or ledger.get("cells") != count
                    or ledger.get("canonical_metric_and_artifact_checks_passed") is not True):
                raise ValueError("Incomplete original task canonical audit")
            identity = expected
        identities.append(identity)
        bound_outputs(Path(source["csv"]).parent, ledger.get("outputs_sha256"))
        if ledger["outputs_sha256"].get(Path(source["csv"]).name) != source["csv_sha256"]:
            raise ValueError("Canonical cell table is not bound by its own ledger")
        plan_path = source["plan"]
        if plan_path not in plans:
            plans[plan_path] = read(plan_path)
    if len(set(identities)) != 4:
        raise ValueError("Duplicate canonical factorial source")
    grids = [p["configs"] for p in plans.values()]
    if not grids or any(grid != grids[0] for grid in grids):
        raise ValueError("Source construction grids differ")
    configs = grids[0]
    expected_factors = set(itertools.product(("wl", "morgan", "path"), (2, 3),
                              ("parity", "minhash", "cw"), ("delta", "union", "delta_union", "roles")))
    factors = [(c["tokenizer"], c["radius"], c["sketch"], c["mode"]) for c in configs.values()]
    if len(configs) != 72 or len(set(factors)) != 72 or set(factors) != expected_factors:
        raise ValueError("The unchanged complete 72-configuration construction grid is required")
    return receipt, directory, configs


def analyze(candidates, task_views, endpoints, families, configs):
    """Freeze one top-three list globally; retain both native/grouped views."""
    names = set(configs)
    required = set(ORDER + ["view", "represented_tasks", "encoding_coverage"])
    required |= {task + suffix for task in TASKS for suffix in ("_rank", "_regret")}
    if not required.issubset(candidates.columns):
        raise ValueError("Missing common-candidate fields")
    if set(candidates.view) != VIEWS or candidates.duplicated(["view", "config"]).any():
        raise ValueError("Missing view or duplicate common candidate")
    for _, group in candidates.groupby("view"):
        if len(group) != 72 or set(group.config) != names or not group.represented_tasks.eq(7).all():
            raise ValueError("Missing configuration or task in common candidate grid")
    numeric = [name for name in required if name not in ("config", "view")]
    if not np.isfinite(candidates[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite candidate diagnostic")
    for table in (task_views, endpoints, families):
        if not {"task", "config"}.issubset(table.columns) or set(table.task) != set(TASKS) or set(table.config) != names:
            raise ValueError("Missing full task coverage in common diagnostic tables")
    if not {"view", "rank", "regret", "encoding_coverage", "source_families"}.issubset(task_views.columns):
        raise ValueError("Missing task-view diagnostic fields")
    if task_views.duplicated(["view", "task", "config"]).any() or set(task_views.view) != VIEWS:
        raise ValueError("Duplicate or missing task-view diagnostic")
    if len(task_views) != 2 * 7 * 72:
        raise ValueError("Incomplete task-view construction grid")
    rankings = []
    for view in sorted(VIEWS):
        ranked = candidates[candidates.view.eq(view)].sort_values(ORDER, kind="stable").copy()
        ranked["global_position"] = np.arange(1, len(ranked) + 1)
        rankings.append(ranked)
    ranking = pd.concat(rankings, ignore_index=True)
    selected = ranking[ranking.view.eq("grouped_main")].head(3).config.tolist()
    positions = {name: i + 1 for i, name in enumerate(selected)}
    ranking["shortlist_position"] = ranking.config.map(positions)
    for name in ("macro_rank", "macro_regret", "worst_task_rank", "encoding_coverage"):
        native = ranking[ranking.view.eq("native_reference")].set_index("config")
        ranking["native_reference_" + name] = ranking.config.map(native[name])
    native_positions = ranking[ranking.view.eq("native_reference")].set_index("config").global_position
    ranking["native_reference_position"] = ranking.config.map(native_positions)
    tables = dict(all_72_rankings=ranking,
                  top3_candidates=ranking[ranking.config.isin(selected)].copy(),
                  top3_task_diagnostics=task_views[task_views.config.isin(selected)].copy(),
                  top3_endpoint_metrics=endpoints[endpoints.config.isin(selected)].copy(),
                  top3_family_diagnostics=families[families.config.isin(selected)].copy())
    for key, table in tables.items():
        if key.startswith("top3"):
            table["shortlist_position"] = table.config.map(positions)
            table.sort_values(["shortlist_position"] + (["view"] if "view" in table else [])
                              + (["task"] if "task" in table else []), inplace=True, kind="stable")
    return selected, tables


def runtime_bindings():
    return {str(path.resolve()): sha(path) for path in
            (HERE / "shortlist.py", HERE / "PROTOCOL.md", HERE / "test_shortlist.py")}


def once(common=COMMON, root=ROOT, output=OUTPUT, common_source=COMMON_SOURCE):
    root, output, common = Path(root), Path(output), Path(common)
    bindings = runtime_bindings()
    completed = root / "COMPLETE.json"
    if completed.exists():
        result = read(completed)
        if (result.get("phase") != "complete" or result.get("configuration_selected") is not None
                or result.get("runtime_bindings") != bindings or result.get("output_root") != str(output.resolve())
                or result.get("common_receipt") != str(common.resolve())):
            raise ValueError("Changed or unrecognized completed shortlist")
        check(common, result["common_receipt_sha256"])
        verify_common(common, common_source)
        bound_outputs(output, result["outputs_sha256"])
        if read(output / "COMPLETE.json") != result:
            raise ValueError("Shortlist completion ledgers differ")
        return result
    if not common.exists():
        status = dict(phase="waiting_for_all_seven", expected_tasks=list(TASKS), configurations=72,
                      configuration_selected=None, refits_started=0, updated_utc=now())
        write(root / "status.json", status)
        write(output / "status.json", status)
        return status
    receipt, directory, configs = verify_common(common, common_source)
    inputs = {name: directory / (name + ".csv") for name in
              ("common_candidates", "task_view_summary", "endpoint_summary", "family_summary")}
    for path in inputs.values():
        if path.name not in receipt["outputs_sha256"]:
            raise ValueError("Required common diagnostic table is not bound")
    selected, tables = analyze(*(pd.read_csv(inputs[name]) for name in
                  ("common_candidates", "task_view_summary", "endpoint_summary", "family_summary")), configs)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name, frame in tables.items():
        target = output / (name + ".csv")
        frame.to_csv(target, index=False)
        files.append(target.name)
    write(output / "shortlist.json", dict(shortlisted_configurations=selected, configuration_selected=None,
          order=ORDER, primary_view="grouped_main", native_view="native_reference",
          selection_scope="three global configurations across all seven tasks",
          interpretation="exploratory_after_prior_test_results_viewed", configs={k: configs[k] for k in selected}))
    files.append("shortlist.json")
    lines = ["# Exploratory shared SynRFP shortlist", "",
        "All 72 constructions cover all seven tasks. The three global candidates are:", ""]
    lines += [str(i) + ". `" + name + "`" for i, name in enumerate(selected, 1)]
    lines += ["", "Ordering: grouped_main macro rank, macro normalized regret, worst task rank, then configuration ID. Each task has equal weight after source-family balancing. Native-reference sensitivity is reported separately.", "",
        "No final preset is selected. These are exploratory diagnostics after test results were inspected; no confidence intervals, significance claims or SOTA claims follow from this shortlist.", "",
        "The completed 72-construction factorial already contains these three candidates on every task. This review starts zero fits and does not repeat those cells. Raw endpoint metrics remain separate; different metrics or regression units are never averaged together.", "",
        "Final baseline comparison remains a separate protocol: audit exact features, row populations, splits, learner settings and retained artifacts before any reuse. Classification factorials contain one grouped five-fold MLP evaluation; the main comparison includes repeated 5×5 grouped CV and kNN. The factorial does not establish completion of those additional fits. Clustering uses five initializations with row-count-based K, distinct from the main known-class-count and adaptive unknown-K studies. Yield keeps the original 64 native partitions; external feasibility uses a fixed expert test. Condition folds average optimizer seeds before fold diagnostics.", "",
        "After user review, freeze a shared preset or a clearly defined three-candidate confirmation protocol. Extend only genuinely missing matched evaluations, preserve published OOD/external tests and account for dependent folds. New seeds on historically viewed data do not erase selection bias. Learned-reference completion is independent of this review.", ""]
    (output / "report.md").write_text("\n".join(lines))
    files.append("report.md")
    # Recheck inputs and producer files before sealing the derived outputs.
    verify_common(common, common_source)
    if bindings != runtime_bindings():
        raise ValueError("Shortlist producer changed during analysis")
    result = dict(schema="synrfp-shared-preset-review-v1", phase="complete", tasks=list(TASKS),
        configurations=72, logical_input_cells=20448, shortlisted_configurations=selected,
        configuration_selected=None, refits_started=0, primary_view="grouped_main", order=ORDER,
        interpretation="exploratory_after_prior_test_results_viewed", test_sets_independent=False,
        common_receipt=str(common.resolve()), common_receipt_sha256=sha(common), runtime_bindings=bindings,
        input_csv_sha256={str(path.resolve()): sha(path) for path in inputs.values()},
        outputs_sha256={name: sha(output / name) for name in files}, output_root=str(output.resolve()),
        finished_utc=now())
    write(output / "COMPLETE.json", result)
    write(root / "COMPLETE.json", result)
    write(root / "status.json", result)
    write(output / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--common", type=Path, default=COMMON)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--poll-seconds", type=int, choices=(30, 60), default=60)
    args = parser.parse_args()
    bindings = runtime_bindings()
    while True:
        if runtime_bindings() != bindings:
            raise ValueError("Shortlist producer changed while waiting")
        result = once(args.common, args.root, args.output)
        print(json.dumps({key: result[key] for key in ("phase", "configuration_selected", "refits_started")}), flush=True)
        if args.once or result["phase"] == "complete":
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
