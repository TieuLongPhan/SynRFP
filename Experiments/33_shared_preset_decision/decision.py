"""A delegated, deferred paper-preset decision from the sealed global shortlist.

This sidecar selects no task-specific winner, launches no fits and changes no
public encoder default. Scientific benchmark reuse requires a separate audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
REVIEW_SOURCE = HERE.parent / "32_shared_preset_review/shortlist.py"
spec = importlib.util.spec_from_file_location("shared_preset33_review", REVIEW_SOURCE)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
ROOT = Path.home() / ".local/share/synrfp-runs/shared_preset_decision_v1"
OUTPUT = HERE / "publication_metadata/shared_preset_decision_v1"
AUTHORITY = {
    "source": "explicit_user_delegation_in_session",
    "date": "2026-10-15",
    "instruction": "I maybe cannot check factorial so maybe you can make decision fo rme",
    "scope": "Choose one shared experimental paper preset across all seven tasks; retain two sensitivity candidates.",
}


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
        raise ValueError("Changed or unbound decision input: " + str(path))


def bindings():
    paths = [HERE / name for name in ("decision.py", "PROTOCOL.md", "test_decision.py")]
    paths += [REVIEW_SOURCE, REVIEW_SOURCE.with_name("PROTOCOL.md"), REVIEW_SOURCE.with_name("test_shortlist.py")]
    return {str(path.resolve()): sha(path) for path in paths}


def choose(receipt, configs, selected, tables):
    """Use the previously fixed global ordering, never per-task maxima."""
    if (receipt.get("phase") != "complete" or receipt.get("configuration_selected") is not None
            or receipt.get("configurations") != 72 or receipt.get("logical_input_cells") != 20448
            or set(receipt.get("tasks", [])) != set(R.TASKS)
            or receipt.get("primary_view") != "grouped_main" or receipt.get("order") != R.ORDER
            or receipt.get("test_sets_independent") is not False):
        raise ValueError("A complete unchanged exploratory seven-task review is required")
    ranking = tables["all_72_rankings"]
    if len(configs) != 72 or set(ranking.config) != set(configs):
        raise ValueError("The complete original 72-configuration grid is required")
    primary = ranking[ranking.view.eq("grouped_main")].sort_values(R.ORDER, kind="stable")
    if (len(primary) != 72 or primary.config.duplicated().any()
            or not primary.represented_tasks.eq(7).all()):
        raise ValueError("Incomplete global ranking")
    expected = primary.head(3).config.tolist()
    if expected != selected or expected != receipt.get("shortlisted_configurations"):
        raise ValueError("The unchanged global shortlist does not match its ranking")
    chosen = expected[0]
    diagnostics = tables["top3_task_diagnostics"]
    own = diagnostics[diagnostics.config.eq(chosen)]
    if set(own.task) != set(R.TASKS) or len(own) != 14:
        raise ValueError("Missing chosen-preset task diagnostics")
    return {
        "configuration_selected": chosen,
        "configuration": configs[chosen],
        "sensitivity_configurations": expected[1:],
        "sensitivity_constructions": {name: configs[name] for name in expected[1:]},
        "selection_rule": {"view": "grouped_main", "order": R.ORDER,
                           "aggregation": "endpoint/regime -> related source family -> equal seven-task weight"},
        "rationale": "First global candidate under the already fixed task-balanced ordering; no task-specific substitutions.",
        "task_diagnostics": own.to_dict(orient="records"),
        "weaknesses": "Task ranks, normalized regrets, encoding coverage and separate raw endpoint metrics remain visible; a global compromise need not win each task.",
    }


def followup(primary):
    """Protocol inventory, deliberately distinct from an executable fit queue."""
    rows = [
        ("classification", 700, 70, "14 endpoints x25 grouped splits x2 learners (MLP and kNN), original n1 policy; factorial has only five MLP folds."),
        ("clustering", 240, 60, "12 endpoints x10 initializations x2 frozen populations (all rows/shared valid); main known-class-count K differs from factorial row-count K. Adaptive unknown-K studies remain separate."),
        ("yield", 64, 64, "64 original partitions: classic BH 14, Suzuki 10, NCS BH 40; preserve official random/OOD regimes."),
        ("barrier", 324, 45, "Primary14 GDB7/Cyclo 110 per preset plus seven15 activation targets 214; preserve random CV, group CV, grouped holdouts, native splits and learning fractions separately. Exclude duplicate15 Cyclo G_act."),
        ("feasibility", 5, 5, "Five optimizer initializations on the same fixed expert test; not five independent tests."),
        ("reagent", 10, 10, "Five author-split initializations plus one five-fold grouped CV."),
        ("condition", 30, 30, "Five author-split initializations plus five fixed grouped folds xfive optimizer seeds; not five repeated CV partitions."),
    ]
    cases = [dict(task=t, primary_preset_protocol_cells=n, factorial_cells_per_configuration=f,
                  protocol=p, reuse_state="requires_exact_identity_audit", reusable_cells_verified=0,
                  missing_cells=None) for t, n, f, p in rows]
    return dict(schema="synrfp-selected-preset-followup-inventory-v1", configuration_selected=primary,
        phase="inventory_requires_compatibility_audit", entries=cases,
        primary_preset_protocol_cells=sum(v[1] for v in rows),
        factorial_cells_per_configuration=sum(v[2] for v in rows),
        new_fits_launched=0, all_benchmarks_complete=False,
        sensitivity_scope="Two retained global candidates already have factorial coverage; no automatic full benchmark expansion.",
        baseline_scope="Reuse existing baseline results only within their exact frozen protocol; this inventory creates no new baseline model jobs.",
        required_identity_checks=["exact encoder construction and version", "feature bytes and failure-vector/validity policy",
            "original row population and ordered IDs", "saved train/validation/calibration/test selectors",
            "learner hyperparameters and numeric backend", "model seeds and threshold/vocabulary fitting scope",
            "retained checkpoint/predictions/metric replay", "full source and artifact hash bindings"],
        specialist_references="Learned yield references continue independently; completed DMPNN/SynCat references retain their scoped datasets. Parrot, T5Chem, adapted feasibility SynCat and full SynCluster are not declared completed or launched here.")


def verify_completed(result, root, output, runtime):
    if (result.get("phase") != "complete" or result.get("runtime_bindings") != runtime
            or result.get("authority") != AUTHORITY or result.get("output_root") != str(output.resolve())):
        raise ValueError("Changed or unrecognized completed delegated decision")
    for path, digest in result["input_sha256"].items():
        check(path, digest)
    R.once()
    for name, digest in result["outputs_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Unexpected decision output path")
        check(output / name, digest)
    if read(output / "COMPLETE.json") != result:
        raise ValueError("Decision completion ledgers differ")
    if read(root / "FROZEN_PRESET.json") != read(output / "FROZEN_PRESET.json"):
        raise ValueError("Frozen preset copies differ")
    return result


def once(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output)
    runtime = bindings()
    completed = root / "COMPLETE.json"
    if completed.exists():
        return verify_completed(read(completed), root, output, runtime)
    # Do not invoke32.once while it is waiting: ownership of that report remains32.
    review_path = R.ROOT / "COMPLETE.json"
    if not review_path.exists():
        status = dict(schema="synrfp-shared-preset-decision-v1", phase="waiting_for_verified_shortlist",
            authority=AUTHORITY, configuration_selected=None, new_fits_launched=0,
            runtime_bindings=runtime, updated_utc=now())
        write(root / "status.json", status)
        write(output / "status.json", status)
        return status
    review = R.once()
    common, directory, configs = R.verify_common(R.COMMON)
    paths = {name: directory / (name + ".csv") for name in
             ("common_candidates", "task_view_summary", "endpoint_summary", "family_summary")}
    selected, tables = R.analyze(*(pd.read_csv(paths[name]) for name in
        ("common_candidates", "task_view_summary", "endpoint_summary", "family_summary")), configs)
    decision = choose(review, configs, selected, tables)
    inputs = {str(path.resolve()): sha(path) for path in [review_path, R.COMMON, *paths.values()]}
    inputs.update(review["input_csv_sha256"])
    frozen = dict(schema="synrfp-frozen-shared-experiment-preset-v1", **decision,
        authority=AUTHORITY, public_api_default_changed=False,
        interpretation="exploratory_after_prior_test_results_viewed", test_sets_independent=False,
        new_seeds_remove_selection_bias=False, source32_configuration_selected=None,
        source_plan_sha256=sorted({s["plan_sha256"] for s in common["sources"]}),
        source_counts={"classification": 5040, "clustering": 4320, "yield": 4608,
                       "barrier_feasibility_reagent_condition": 6480, "total": 20448},
        construction_scope="One unchanged 2048-bit original-grid construction; no new roles variant or stereo extension.",
        frozen_utc=now())
    output.mkdir(parents=True, exist_ok=True)
    endpoints = tables["top3_endpoint_metrics"]
    endpoints[endpoints.config.eq(decision["configuration_selected"])].to_csv(output / "selected_endpoint_metrics.csv", index=False)
    write(output / "FROZEN_PRESET.json", frozen)
    write(root / "FROZEN_PRESET.json", frozen)
    write(output / "decision.json", dict(**frozen, new_fits_launched=0,
        all_benchmarks_complete=False, raw_endpoint_metrics="selected_endpoint_metrics.csv"))
    write(output / "benchmark_followup.json", followup(decision["configuration_selected"]))
    lines = ["# Delegated shared experimental preset", "",
        "Selected: `" + decision["configuration_selected"] + "`.", "",
        "The user delegated this choice on 15 October 2026. The first grouped_main global candidate is frozen using the existing ordering: macro rank, macro normalized regret, worst task rank, configuration ID. Each task has equal weight after source-family balancing. No different preset is chosen for an individual task.", "",
        "Sensitivity candidates: " + ", ".join("`" + n + "`" for n in selected[1:]) + ".", "",
        "This freezes an experimental paper preset; it changes no public API default. The exact native construction and per-task weaknesses are recorded in FROZEN_PRESET.json; raw endpoint metrics retain their original units in selected_endpoint_metrics.csv.", "",
        "Selection used historically inspected tests and remains exploratory. Additional random seeds do not remove this selection bias. It provides no new SOTA, significance or independent-test claim. Study32 retains its original configuration_selected=null record; this separate authorized decision does not rewrite historical ledgers.", "",
        "benchmark_followup.json inventories 1373 primary-preset protocol cells; the number of missing fits remains unknown. Zero cells are declared reusable without exact feature, row, selector, learner and artifact checks. Factorial coverage is not equivalent to every main benchmark: classification needs repeated splits and kNN; clustering uses different K and repetition settings. No new fit is launched by this decision. Existing GPU queues continue independently.", ""]
    (output / "report.md").write_text("\n".join(lines))
    R.once()
    for path, digest in inputs.items():
        check(path, digest)
    if runtime != bindings():
        raise ValueError("Decision producer changed during selection")
    names = ["FROZEN_PRESET.json", "decision.json", "benchmark_followup.json", "selected_endpoint_metrics.csv", "report.md"]
    result = dict(schema="synrfp-shared-preset-decision-v1", phase="complete", authority=AUTHORITY,
        configuration_selected=decision["configuration_selected"], sensitivity_configurations=selected[1:],
        source32_configuration_selected=None, runtime_bindings=runtime, input_sha256=inputs,
        outputs_sha256={name: sha(output / name) for name in names},
        output_root=str(output.resolve()), new_fits_launched=0, all_benchmarks_complete=False,
        interpretation="exploratory_after_prior_test_results_viewed", finished_utc=now())
    write(output / "COMPLETE.json", result)
    write(completed, result)
    write(root / "status.json", result)
    write(output / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    # This owned controller lock is persistent: never replace or unlink it.
    with (ROOT / "controller.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = bindings()
        while True:
            if runtime != bindings():
                raise ValueError("Decision or reviewed producer changed while waiting")
            result = once()
            print(json.dumps(result), flush=True)
            if args.once or result["phase"] == "complete":
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
