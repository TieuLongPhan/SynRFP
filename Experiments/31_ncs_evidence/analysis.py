"""Read-only, provenance-bound explanatory analysis of completed factorials.

This worker trains no models and selects no fingerprint preset. Its completion
marker concerns the seven-task descriptive analysis, never all reference models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = Path.home() / ".local/share/synrfp-runs/ncs_evidence_v1"
OUTPUT = HERE / "publication_metadata/ncs_evidence_v1"
OLD_ROOT = Path.home() / ".local/share/synrfp-runs/all_endpoint_factorial_full_v1"
OLD_EXPORT = EXPERIMENTS / "22_all_endpoint_factorial/publication_metadata/all_endpoint_factorial_full_v1"
NEW_ROOT = Path.home() / ".local/share/synrfp-runs/cross_task_factorial_v1"
NEW_EXPORT = EXPERIMENTS / "30_cross_task_factorial/publication_metadata/cross_task_factorial_v1"
TASKS = ("classification", "clustering", "yield", "barrier", "feasibility", "reagent", "condition")
COUNTS = {"classification": 5040, "clustering": 4320, "yield": 4608}
METRICS = dict(classification="mcc", clustering="ami", **{"yield": "r2"},
               barrier="rmse", feasibility="auroc", reagent="micro_f1", condition="micro_f1")
FACTORS = ["tokenizer", "radius", "sketch"]
CONTRASTS = (("delta_union", "delta"), ("roles", "delta"),
             ("roles", "delta_union"), ("union", "delta"))
UNIT = ["task", "source", "family", "endpoint", "regime", "partition"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def check_hash(path, expected):
    if not expected or sha(path) != expected:
        raise ValueError(f"Changed or unbound analysis input: {path}")


def verified_cells(table, receipt_path, plan_path, expected, task=None):
    """Reject incomplete ledgers and verify every bound report output, not just CSV."""
    table, receipt_path, plan_path = map(Path, (table, receipt_path, plan_path))
    receipt = read(receipt_path)
    if receipt.get("phase") != "complete" or receipt.get("plan_sha256") != sha(plan_path):
        raise ValueError("Incomplete or changed factorial completion ledger")
    if task is not None:
        if (receipt.get("task") != task or receipt.get("cells") != expected
                or receipt.get("canonical_metric_and_artifact_checks_passed") is not True
                or receipt.get("configuration_selected") is not None):
            raise ValueError("Unverified task completion ledger")
    else:
        if (receipt.get("completed") != expected or receipt.get("expected") != expected
                or receipt.get("canonical_metrics_and_artifacts_audited") != expected):
            raise ValueError("Incomplete four-task canonical audit")
    outputs = receipt.get("outputs_sha256", {})
    if table.name not in outputs:
        raise ValueError("Canonical cells were not bound by the completion ledger")
    for name, digest in outputs.items():
        if Path(name).name != name:
            raise ValueError("Unexpected output path in factorial completion ledger")
        check_hash(table.parent / name, digest)
    frame = pd.read_csv(table)
    if len(frame) != expected or "phase" in frame and not frame.phase.eq("complete").all():
        raise ValueError("Missing or unfinished canonical cells")
    identity = dict(csv=str(table.resolve()), csv_sha256=sha(table),
                    complete=str(receipt_path.resolve()), complete_sha256=sha(receipt_path),
                    plan=str(plan_path.resolve()), plan_sha256=sha(plan_path), cells=expected)
    return frame, identity


def family(task, source):
    if task in ("classification", "clustering"):
        for prefix in ("schneider", "tpl", "uspto50k", "syntemp", "ecreact", "mech"):
            if source.startswith(prefix):
                return prefix
        raise ValueError("Unknown classification/clustering family: " + source)
    if task == "barrier":
        if source in ("gdb7", "rdb7"):
            return "gdb7_rdb7_related"
        if source in ("e2", "sn2", "e2sn2"):
            return "e2_sn2_related"
    return source


def normalize_old(frame, task):
    source = frame["source_dataset"] if "source_dataset" in frame else frame["dataset"]
    out = pd.DataFrame(dict(task=task, source=source.astype(str), endpoint=frame.dataset.astype(str),
                            config=frame.config.astype(str), score=frame[METRICS[task]],
                            seed=frame.seed.astype(str)))
    if task == "classification":
        if "family" in frame and not frame.family.eq("classification").all():
            raise ValueError("EHReact activity must remain outside the seven-task report")
        out["regime"] = "grouped_cv"
        out["partition"] = frame.fold.astype(str)
        out["repeat_kind"] = "heldout_group_fold"
    elif task == "clustering":
        out["regime"] = "heuristic_rowcount_kmeans"
        # Initialization variation is preserved and explicitly never called CV.
        out["partition"] = "cluster_init_" + frame.seed.astype(str)
        out["repeat_kind"] = "cluster_initialization"
    else:
        split = frame.dataset.astype(str).str.split("__", n=1)
        if not split.map(len).eq(2).all():
            raise ValueError("Yield source lacks its native evaluation regime")
        out["regime"] = split.str[1]
        out["partition"] = frame.split.astype(str)
        out["seed"] = "42"
        out["repeat_kind"] = "native_data_partition"
    out["family"] = [family(task, value) for value in out.source]
    return out


def normalize_new(frame):
    if set(frame.task) != {"barrier", "feasibility", "reagent", "condition"}:
        raise ValueError("New canonical cells must cover exactly four declared tasks")
    rows = []
    for r in frame.to_dict("records"):
        task, source, split = str(r["task"]), str(r["source"]), str(r["split"])
        raw = str(r.get("protocol") or r.get("panel") or "")
        if task == "feasibility":
            regime, partition, kind = "external_fixed_test", "one_expert_test_population", "optimizer_seed_fixed_test"
        elif raw in ("native", "native_published", "author_native") or split.startswith("native"):
            regime, partition, kind = "native", "one_author_test_population", "optimizer_seed_fixed_test"
        elif raw in ("group_disjoint", "group_cv", "reactant_direction_group_kfold", "reactant_group_kfold") or split.startswith(("group_fold", "group_cv")):
            regime, kind = "grouped_cv", "heldout_group_fold"
            if task == "condition":
                fold = r.get("fold")
                partition = str(int(fold)) if fold is not None and not pd.isna(fold) else split.split("_seed", 1)[0]
            else:
                partition = split
        elif raw == "reactant_group_holdout" or split.startswith("reactant_group_"):
            regime, partition, kind = "grouped_holdout", split, "repeated_group_holdout"
        else:
            raise ValueError(f"Unknown task evaluation regime: {task}/{raw}/{split}")
        target = r.get("target", "")
        target = "" if pd.isna(target) else str(target)
        rows.append(dict(task=task, source=source, family=family(task, source),
                         endpoint=source + "__" + target, regime=regime, partition=partition,
                         repeat_kind=kind, seed=str(r["seed"]), config=str(r["config"]), score=float(r[METRICS[task]])))
    return pd.DataFrame(rows)


def validate_grid(cells, configs):
    names = set(configs)
    if len(names) != 72:
        raise ValueError("The evidence analysis requires the frozen 72-configuration grid")
    factors = pd.DataFrame.from_dict(configs, orient="index")
    if (set(factors.tokenizer) != {"wl", "morgan", "path"} or set(factors.radius) != {2, 3}
            or set(factors.sketch) != {"parity", "minhash", "cw"}
            or set(factors["mode"]) != {"delta", "union", "delta_union", "roles"}
            or factors.duplicated(FACTORS + ["mode"]).any()):
        raise ValueError("Changed factorial construction grid")
    if not np.isfinite(cells.score.to_numpy(float)).all():
        raise ValueError("Missing or nonfinite canonical metric")
    if set(cells.config) != names or cells.duplicated(UNIT + ["seed", "config"]).any():
        raise ValueError("Missing configuration or duplicate paired cell")
    for _, group in cells.groupby(UNIT, dropna=False):
        if set(group.config) != names:
            raise ValueError("Incomplete paired construction grid at a native partition")
        seeds = group.groupby("config").seed.apply(lambda x: frozenset(x))
        if len(set(seeds)) != 1:
            raise ValueError("Optimizer seed populations differ between paired modes")


def analyze(cells, configs):
    validate_grid(cells, configs)
    # Five condition optimizer initializations form one fold mean. Native/FREA
    # seeds similarly share one test population and do not create extra folds.
    units = cells.groupby(UNIT + ["repeat_kind", "config"], dropna=False).agg(
        score=("score", "mean"), optimizer_seed_sd=("score", "std"),
        raw_cells=("score", "size")).reset_index()
    for name in FACTORS + ["mode"]:
        units[name] = units.config.map({key: value[name] for key, value in configs.items()})
    paired = []
    keys = UNIT + ["repeat_kind"] + FACTORS
    for identities, group in units.groupby(keys, dropna=False):
        modes = group.set_index("mode")
        if set(modes.index) != {"delta", "union", "delta_union", "roles"}:
            raise ValueError("Unpaired modes after fold aggregation")
        context = dict(zip(keys, identities))
        for left, right in CONTRASTS:
            a, b = modes.loc[left], modes.loc[right]
            difference = float(a.score - b.score)
            paired.append(dict(**context, contrast=left + "_minus_" + right,
                left_mode=left, right_mode=right, left_score=float(a.score), right_score=float(b.score),
                difference=difference, benefit=-difference if context["task"] == "barrier" else difference,
                metric=METRICS[context["task"]], left_raw_cells=int(a.raw_cells), right_raw_cells=int(b.raw_cells)))
    matched = pd.DataFrame(paired)
    endpoint_keys = ["task", "source", "family", "endpoint", "regime"] + FACTORS + ["contrast", "metric"]
    endpoints = matched.groupby(endpoint_keys, dropna=False).agg(
        mean_difference=("difference", "mean"), descriptive_partition_sd=("difference", "std"),
        mean_benefit=("benefit", "mean"), paired_partitions=("difference", "size"),
        positive_partitions=("benefit", lambda x: int((x > 0).sum()))).reset_index()
    family_keys = ["task", "family", "regime"] + FACTORS + ["contrast", "metric"]
    families = endpoints.groupby(family_keys, dropna=False).agg(
        mean_difference=("mean_difference", "mean"), mean_benefit=("mean_benefit", "mean"),
        endpoints=("endpoint", "nunique")).reset_index()
    # Native and grouped condition/reagent views remain separate. Within each
    # view, regimes are averaged inside a source family before family balancing.
    tasks = []
    for view, regime in (("grouped_main", "grouped_cv"), ("native_reference", "native")):
        chosen = families[(~families.task.isin(["reagent", "condition"])) | families.regime.eq(regime)]
        per_family = chosen.groupby(["task", "family"] + FACTORS + ["contrast", "metric"], dropna=False).agg(
            mean_difference=("mean_difference", "mean"), mean_benefit=("mean_benefit", "mean"),
            regimes=("regime", "nunique")).reset_index()
        per_task = per_family.groupby(["task"] + FACTORS + ["contrast", "metric"], dropna=False).agg(
            mean_difference=("mean_difference", "mean"), mean_benefit=("mean_benefit", "mean"),
            source_families=("family", "nunique")).reset_index()
        per_task["view"] = view
        tasks.append(per_task)
    return dict(matched_mode_contrasts=matched, endpoint_contrasts=endpoints,
                family_contrasts=families, task_contrasts=pd.concat(tasks, ignore_index=True))


def verify_common(path):
    receipt = read(path)
    if (receipt.get("phase") != "complete" or receipt.get("configuration_selected") is not None
            or receipt.get("logical_input_cells") != 20448 or set(receipt.get("tasks", [])) != set(TASKS)
            or receipt.get("configurations") != 72):
        raise ValueError("Incomplete or changed seven-task common-analysis receipt")
    if len(receipt.get("sources", [])) != 4:
        raise ValueError("Common analysis must bind exactly four canonical factorial ledgers")
    for source in receipt["sources"]:
        for name, digest in (("csv", "csv_sha256"), ("complete", "complete_sha256"), ("plan", "plan_sha256")):
            check_hash(source[name], source[digest])
    for name, digest in receipt.get("outputs_sha256", {}).items():
        if Path(name).name != name:
            raise ValueError("Unexpected common-analysis output name")
        check_hash(Path(receipt["output_root"]) / name, digest)
    return receipt


def inventory():
    """Record existing evidence bytes; this is not a rerun or an identity proof."""
    base = EXPERIMENTS / "12_validation/runs"
    groups = {
        "cluster_coherence": list((base / "cluster_coherence/n1/all_rows").glob("*/*/*.json")),
        "representative_selection": list((base / "ncs_extensions/representative_selection").glob("*/*.json")),
        "clustering_attribution": list((base / "clustering_attribution").rglob("summary.csv")),
        "locality_collisions": [base / "locality" / n for n in ("collisions.csv", "bounds_summary.csv", "bounds_class.csv")],
        "cost": [base / "cost" / n for n in ("encoding.csv", "encoding_rxnfp.csv", "search.csv", "hardware.json")],
        "bond_coherence": [base / "ncs_extensions/bond_coherence" / n for n in ("summary.csv", "coverage.json")],
    }
    records = []
    for group, paths in groups.items():
        for path in sorted(paths):
            records.append(dict(group=group, path=str(path.resolve()), present=path.is_file(),
                                bytes=path.stat().st_size if path.is_file() else 0,
                                sha256=sha(path) if path.is_file() else ""))
    frame = pd.DataFrame(records)
    summary = dict(scope="Read-only inventory with hashes at analysis time; historical scores are not rerun",
                   files_by_group={key: sum(p.is_file() for p in paths) for key, paths in groups.items()},
                   crxnfp_in_legacy_utility=False, legacy_utility_rows_per_dataset=20000,
                   legacy_utility_seeds=3, legacy_utility_budgets=[0.005, 0.01, 0.02, 0.05],
                   legacy_synrfp="wl/r2/1024/parity/delta; different from the new 2048-bit grid",
                   limitations=["No crxnfp comparator in legacy chemical utility or bond-coherence tables",
                     "ECReact L1 and USPTO balanced class coverage saturates; no general utility dominance established",
                     "Product-group coverage is not an independent transformation identity or experimental usefulness measure",
                     "Legacy cost uses different settings/hardware/load; current native fit times are descriptive, not a fair speed ranking",
                     "Legacy locality/collisions describe WL radius2 delta at1024bits, not every new mode or sketch",
                     "No learned-reference completion is inferred by this analysis"])
    utility = [read(path) for path in groups["representative_selection"]]
    if utility:
        u = pd.DataFrame(utility)
        expected = 2 * 7 * 3 * 4
        summary["utility_expected_cells"] = expected
        summary["utility_observed_cells"] = len(u)
        summary["utility_complete_inventory"] = bool(len(u) == expected and not u.duplicated(["dataset", "fingerprint", "budget", "seed"]).any())
        summary["utility_fingerprints"] = sorted(u.fingerprint.unique().tolist())
        means = u.groupby(["dataset", "fingerprint", "budget"])[["class_coverage", "product_group_coverage"]].mean().reset_index()
        summary["utility_descriptive_means"] = means.to_dict("records")
    bond = base / "ncs_extensions/bond_coherence/summary.csv"
    if bond.exists():
        b = pd.read_csv(bond)
        summary["bond_coherence_cells"] = len(b)
        summary["bond_coherence_proxy_means"] = b.groupby("fingerprint")[["signature_agreement", "signature_agreement_different_product_group"]].mean().reset_index().to_dict("records")
    return frame, summary


def optional_status(root, output):
    paths = [root / "token_audit/status.json", root / "token_audit/COMPLETE.json",
             output / "token_audit/status.json", output / "token_audit/COMPLETE.json",
             EXPERIMENTS / "17_yield_baselines/publication_metadata/matched_full_v3/unattended_v4/status.json"]
    return [{"path": str(p.resolve()), "sha256": sha(p), "status": read(p)} for p in paths if p.is_file()]


def file_signature(path):
    """Cheap cache invalidation; changed files still require full hash admission."""
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, inode=stat.st_ino)


def run(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output)
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    common_path = NEW_ROOT / "COMMON_TASK_ANALYSIS_COMPLETE.json"
    optional = optional_status(root, output)
    receipts = [OLD_EXPORT / task / "COMPLETE.json" for task in COUNTS]
    watched = {OLD_ROOT / "plan.json"}
    for receipt_path in receipts:
        watched.update(receipt_path.parent / name for name in read(receipt_path).get("outputs_sha256", {}))
    if common_path.exists():
        record = read(common_path)
        for source in record.get("sources", []):
            watched.update(Path(source[name]) for name in ("csv", "complete", "plan"))
        watched.update(Path(record["output_root"]) / name for name in record.get("outputs_sha256", {}))
    signature = dict(script=sha(__file__), protocol=sha(HERE / "PROTOCOL.md"),
                     receipts={str(p.resolve()): sha(p) for p in receipts},
                     common=sha(common_path) if common_path.exists() else None,
                     watched_file_metadata=[file_signature(p) for p in sorted(watched)],
                     optional={r["path"]: r["sha256"] for r in optional})
    previous = root / "analysis_state.json"
    if previous.exists():
        state = read(previous)
        if state.get("input_signature") == signature:
            for name, digest in state["outputs_sha256"].items():
                check_hash(output / name, digest)
            return dict(state["status"], idempotent_reuse=True)
    old_plan = read(OLD_ROOT / "plan.json")
    configs = old_plan["configs"]
    frames, sources = [], []
    for task, expected in COUNTS.items():
        frame, identity = verified_cells(OLD_EXPORT / task / "cells.csv", OLD_EXPORT / task / "COMPLETE.json",
                                         OLD_ROOT / "plan.json", expected, task)
        frames.append(normalize_old(frame, task))
        sources.append(identity)
    common = None
    if common_path.exists():
        common = verify_common(common_path)
        plan = read(NEW_ROOT / "plan.json")
        if plan["configs"] != configs:
            raise ValueError("The new and old canonical construction grids differ")
        frame, identity = verified_cells(NEW_EXPORT / "cells.csv", NEW_ROOT / "COMPLETE.json", NEW_ROOT / "plan.json", 6480)
        if not any(s["csv_sha256"] == identity["csv_sha256"] and s["complete_sha256"] == identity["complete_sha256"] for s in common["sources"]):
            raise ValueError("Four-task canonical cells differ from common-analysis attestation")
        frames.append(normalize_new(frame))
        sources.append(identity)
    cells = pd.concat(frames, ignore_index=True)
    ready = [task for task in TASKS if task in set(cells.task)]
    outputs = analyze(cells, configs)
    evidence, evidence_summary = inventory()
    inputs = dict(factorial_sources=sources, common_analysis_receipt=str(common_path) if common else None,
                  common_analysis_sha256=sha(common_path) if common else None,
                  optional_statuses=optional, input_signature=signature)
    for name, frame in outputs.items():
        csv(output / (name + ".csv"), frame)
    csv(output / "evidence_inventory.csv", evidence)
    write(output / "evidence_summary.json", evidence_summary)
    write(output / "inputs.json", inputs)
    full = set(ready) == set(TASKS)
    status = dict(schema="synrfp-ncs-evidence-v1", phase="complete" if full else "waiting_for_all_seven",
                  ready_tasks=ready, expected_tasks=7, canonical_input_cells=len(cells),
                  paired_mode_contrasts=len(outputs["matched_mode_contrasts"]), configuration_selected=None,
                  analysis_scope="Seven-task descriptive explanation only; not completion of learned references or all NCS work",
                  preliminary=not full, independent_test_sets=False, updated_utc=now())
    lines = ["# NCS evidence analysis", "", "All seven task factorials are available." if full else
        "PRELIMINARY: classification, clustering and yield only. Four additional task factorials are pending.", "",
        f"Canonical input cells: {len(cells):,}. Ready tasks: {', '.join(ready)}. No configuration is selected.", "",
        "The four prespecified contrasts are delta_union minus delta, roles minus delta, roles minus delta_union, and union minus delta. "
        "Each pair matches tokenizer, radius, sketch, source endpoint, evaluation regime and native partition. "
        "Optimizer seeds sharing a condition fold or fixed native/expert test population are averaged before the contrast. "
        "Clustering initialization seeds remain explicitly labelled initialization comparisons, never CV folds.", "",
        "Endpoint effects average partition contrasts; families average related endpoints; task summaries balance families. "
        "Grouped and native reagent/condition views remain separate. Barrier benefit negates the raw RMSE difference; "
        "other benefits equal raw score differences. Metrics are never averaged across tasks. "
        "Standard deviations are descriptive correlated-partition variation, with no p-values or confidence claims.", "",
        "## NCS priorities after the current queue", "",
        "1. Explain the balance between changed and retained context using the matched mode effects and the separate bounded token audit. "
        "At full token-count level U=|D|+2S, so delta_union and roles encode equivalent count information before support/sketch compression; "
        "a gain between their finite fingerprints needs a compression or budget explanation.",
        "2. Preserve clustering as a central application, with known-resolution and label-free resolution experiments kept distinct. "
        "Legacy chemical coherence is useful evidence; representative selection has mixed outcomes and must not be sold as universal superiority. "
        "Any further utility study needs a new fixed protocol and the missing crxnfp comparator.",
        "3. Jointly choose one practical preset after this diagnostic table; no automatic selection or test-max promotion occurs here. "
        "Treat historical-data selection as exploratory, and separate later confirmation from this screen.",
        "4. Finish specialist references already queued, report barrier weaknesses and compute tradeoffs, then decide which additional references justify effort. "
        "This report does not train Parrot, T5Chem, adapted SynCat feasibility or a full SynCluster comparison.",
        "5. Use native recorded fit times and cache sizes as descriptive cost evidence now. "
        "Run an isolated fixed-hardware encoding benchmark only after the busy research queues drain.", "",
        "## Existing evidence and boundaries", "",
        f"Inventory: {len(evidence)} files with SHA-256 values, read without modifying original reports, feature caches or model weights.", "",
        "Legacy SynRFP utility/locality uses WL radius2 delta at1024bits; it does not establish new 2048-bit modes or all sketches. "
        "The 168-cell representative-selection study uses two 20,000-row samples, three seeds, four budgets, five fingerprints, random and greedy controls. "
        "It lacks crxnfp and includes rapidly saturating coarse-label coverage. "
        "Product-group diversity and mapped element/bond-change signatures are declared proxies, not exact reaction identity or experimental utility.", "",
        "Task and source-family counts are aggregation conventions, not claims that all datasets are independent. "
        "EHReact is excluded. Existing negative results remain visible. NCS suitability is a research target, not an acceptance prediction.", ""]
    (output / "report.md").write_text("\n".join(lines))
    write(output / "status.json", status)
    names = [name + ".csv" for name in outputs] + ["evidence_inventory.csv", "evidence_summary.json", "inputs.json", "report.md", "status.json"]
    hashes = {name: sha(output / name) for name in names}
    receipt = dict(**status, outputs_sha256=hashes, source_sha256=sha(__file__), protocol_sha256=sha(HERE / "PROTOCOL.md"),
                   factorial_sources=sources, learned_reference_completion_asserted=False)
    if full:
        write(output / "COMPLETE.json", receipt)
        hashes["COMPLETE.json"] = sha(output / "COMPLETE.json")
    write(root / "analysis_state.json", dict(input_signature=signature, outputs_sha256=hashes, status=status))
    write(root / "run_manifest.json", dict(source_sha256=sha(__file__), protocol_sha256=sha(HERE / "PROTOCOL.md"),
                                            factorial_sources=sources, output=str(output.resolve()), updated_utc=now()))
    return status


def producer_receipt(root=ROOT, output=OUTPUT):
    """An attempt-scoped routing receipt; environment values never affect math."""
    stage_id = os.environ.get("NCS_EVIDENCE_STAGE_ID")
    input_key = os.environ.get("NCS_EVIDENCE_INPUT_KEY")
    if stage_id is None and input_key is None:
        return
    if not stage_id or not re.fullmatch(r"[0-9a-f]{32}", stage_id) or not input_key:
        raise ValueError("Missing or invalid evidence producer routing identity")
    result = Path(output).resolve() / "status.json"
    write(Path(root) / "stages" / stage_id / "PRODUCER_READY.json", dict(
        phase="complete", stage_id=stage_id, input_key=input_key,
        producer_sha256=sha(__file__), result_path=str(result), result_sha256=sha(result)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Produce or reuse one bounded analysis; never wait or train")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.root, args.output)
    producer_receipt(args.root, args.output)
    print(json.dumps(result, indent=2))
