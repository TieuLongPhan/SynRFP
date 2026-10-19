"""Exploratory, task-balanced candidate diagnostics after all factorials close.

This report reads signed canonical cell tables; it never fits a model, changes
an encoder, selects a preset or promotes a test maximum as confirmation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cross_task30_common_report_study", HERE / "study.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
OLD_ROOT = Path.home() / ".local/share/synrfp-runs/all_endpoint_factorial_full_v1"
OLD_EXPORT = HERE.parent / "22_all_endpoint_factorial/publication_metadata/all_endpoint_factorial_full_v1"
TASKS = ("classification", "clustering", "yield", "barrier", "feasibility", "reagent", "condition")
METRICS = {"classification": "mcc", "clustering": "ami", "yield": "r2", "barrier": "rmse",
           "feasibility": "auroc", "reagent": "micro_f1", "condition": "micro_f1"}
OLD_COUNTS = dict(classification=5040, clustering=4320, **{"yield": 4608})
GROUP_ALIASES = {"native": "native", "native_published": "native", "author_native": "native",
                 "group_disjoint": "grouped_cv", "group_cv": "grouped_cv",
                 "reactant_direction_group_kfold": "grouped_cv", "reactant_group_kfold": "grouped_cv",
                 "reactant_group_holdout": "grouped_holdout", "external_fixed_test": "external_fixed_test"}


def read(path):
    return json.loads(Path(path).read_text())


def canonical_cells(csv_path, receipt_path, plan_path, expected, task=None, new=False):
    """Bind the completed canonical ledger and exact CSV bytes before reading."""
    csv_path, receipt_path, plan_path = map(Path, (csv_path, receipt_path, plan_path))
    receipt = read(receipt_path)
    if receipt.get("phase") != "complete" or receipt.get("plan_sha256") != S.sha(plan_path):
        raise ValueError("Incomplete or changed source factorial ledger")
    if new:
        plan = read(plan_path)
        if (receipt.get("completed") != expected or receipt.get("expected") != expected
                or receipt.get("reused") != plan["expected_reused_models"]
                or receipt.get("canonical_metrics_and_artifacts_audited") != expected):
            raise ValueError("Incomplete four-task canonical audit count")
    elif (receipt.get("task") != task or receipt.get("cells") != expected
          or receipt.get("canonical_metric_and_artifact_checks_passed") is not True):
        raise ValueError("Incomplete original task canonical audit")
    expected_hash = receipt.get("outputs_sha256", {}).get(csv_path.name)
    if not expected_hash or S.sha(csv_path) != expected_hash:
        raise ValueError("Changed canonical factorial cell table")
    frame = pd.read_csv(csv_path)
    if len(frame) != expected or "phase" in frame and not frame.phase.eq("complete").all():
        raise ValueError("Canonical table has missing or unfinished cells")
    provenance = dict(csv=str(csv_path.resolve()), csv_sha256=expected_hash,
                      complete=str(receipt_path.resolve()), complete_sha256=S.sha(receipt_path),
                      plan=str(plan_path.resolve()), plan_sha256=S.sha(plan_path), cells=expected)
    return frame, provenance


def family(task, source):
    source = str(source)
    if task in ("classification", "clustering"):
        for prefix in ("schneider", "uspto50k", "syntemp", "ecreact", "mech", "tpl"):
            if source.startswith(prefix):
                return prefix
        raise ValueError("Undeclared classification/clustering source family: " + source)
    if task == "yield":
        if source not in ("bh_classic", "suzuki_hte", "ncs_bh"):
            raise ValueError("Undeclared yield source family")
        return source
    if task == "barrier":
        if source in ("gdb7", "rdb7"):
            return "gdb7_rdb7_related"
        if source in ("e2", "sn2", "e2sn2"):
            return "e2_sn2_related"
        return source
    return source


def normalize_regime(value, split=""):
    value = str(value)
    if value in GROUP_ALIASES:
        return GROUP_ALIASES[value]
    if str(split).startswith("native"):
        return "native"
    if str(split).startswith(("group_fold", "group_cv")):
        return "grouped_cv"
    if str(split).startswith("reactant_group_"):
        return "grouped_holdout"
    if value.startswith("seed"):
        return "external_fixed_test"
    raise ValueError("Undeclared evaluation regime: " + value)


def old_normalized(frame, task):
    out = pd.DataFrame(dict(task=task, config=frame["config"], source=frame.get("source_dataset", frame["dataset"]),
                            endpoint=frame["dataset"], score=frame[METRICS[task]], seed=frame["seed"].astype(str)))
    if task == "classification":
        if "family" in frame and not frame.family.eq("classification").all():
            raise ValueError("EHReact activity must remain excluded")
        out["regime"] = "grouped_cv"
        out["fold"] = frame["fold"].astype(str)
    elif task == "clustering":
        out["regime"] = "heuristic_rowcount_kmeans"
        out["fold"] = "one_input_population"
    else:
        out["regime"] = frame.dataset.str.split("__", n=1).str[1]
        out["fold"] = frame["split"].astype(str)
        # Existing yield fits vary their data partition, not model initialization.
        out["seed"] = "42"
    if task == "clustering":
        out["encoding_coverage"] = frame["shared_valid_coverage"]
        out["coverage_scope"] = "full_original_clustering_population"
    else:
        out["encoding_coverage"] = 1 - frame["n_unfeaturised_test"] / frame["n_test"]
        out["coverage_scope"] = "original_heldout_population"
    out["test_rows"] = frame["n"] if task == "clustering" else frame["n_test"]
    out["origin"] = "study22_canonical_cuda" if task != "clustering" else "study22_canonical_cpu"
    out["family"] = [family(task, source) for source in out.source]
    return out


def new_normalized(frame):
    expected_tasks = {"barrier", "feasibility", "reagent", "condition"}
    if set(frame.task) != expected_tasks:
        raise ValueError("New factorial table must contain exactly its four declared tasks")
    rows = []
    for _, value in frame.iterrows():
        task, source = str(value["task"]), str(value["source"])
        split = str(value["split"])
        raw_regime = value.get("protocol", value.get("panel", ""))
        if pd.isna(raw_regime) or not str(raw_regime):
            raw_regime = value.get("panel", "")
        regime = "external_fixed_test" if task == "feasibility" else normalize_regime(raw_regime, split)
        if task == "feasibility":
            fold = "one_expert_test_population"
        elif regime == "native":
            fold = "one_author_test_population"
        elif task == "condition":
            raw_fold = value.get("fold")
            if pd.isna(raw_fold):
                raw_fold = split.split("_seed", 1)[0]
            fold = str(raw_fold)
        else:
            fold = split
        target = str(value.get("target", ""))
        rows.append(dict(task=task, source=source, family=family(task, source),
            endpoint=source + "__" + target, regime=regime, config=value["config"],
            fold=fold, seed=str(value["seed"]), score=float(value[METRICS[task]]),
            encoding_coverage=float(value["encoding_valid_fraction"]),
            coverage_scope="frozen_original_task_population",
            test_rows=int(value.get("n_test", value.get("test_rows", 751 if task == "feasibility" else 0))),
            origin=value["origin"]))
    return pd.DataFrame(rows)


def validate_grid(frame, configs):
    configs = set(configs)
    if set(frame.task) != set(TASKS) or set(frame.config) != configs:
        raise ValueError("The common report requires all seven tasks and the exact construction grid")
    if not np.isfinite(frame.score.to_numpy(float)).all():
        raise ValueError("Missing or nonfinite primary metric")
    if not np.isfinite(frame.encoding_coverage.to_numpy(float)).all() or not frame.encoding_coverage.between(0, 1).all():
        raise ValueError("Invalid canonical encoding coverage")
    identity = ["task", "source", "endpoint", "regime", "config", "fold", "seed"]
    if frame.duplicated(identity).any():
        raise ValueError("Duplicate canonical evaluation cell")
    keys = ["task", "source", "endpoint", "regime", "fold"]
    for _, group in frame.groupby(keys, dropna=False):
        if set(group.config) != configs:
            raise ValueError("Missing construction at an endpoint/partition")
        seeds = group.groupby("config").seed.apply(lambda x: frozenset(x))
        if len(set(seeds)) != 1:
            raise ValueError("Construction seed populations differ")


def analyze(cells, configs=None):
    """Separate optimizer and fold variation before balancing source families."""
    configs = S.CONFIGS if configs is None else configs
    cells = cells.copy()
    validate_grid(cells, configs)
    base = ["task", "source", "family", "endpoint", "regime", "config"]
    folds = cells.groupby(base + ["fold"], dropna=False).agg(
        score=("score", "mean"), within_seed_sd=("score", "std"),
        optimizer_seeds=("seed", "nunique"), test_rows=("test_rows", "mean"),
        encoding_coverage=("encoding_coverage", "mean"), cells=("score", "size")).reset_index()
    endpoints = folds.groupby(base, dropna=False).agg(
        mean_score=("score", "mean"), between_partition_sd=("score", "std"),
        mean_within_seed_sd=("within_seed_sd", "mean"), partitions=("fold", "size"),
        optimizer_seeds_per_partition=("optimizer_seeds", "mean"),
        repeated_test_rows_sum=("test_rows", "sum"), encoding_coverage=("encoding_coverage", "mean"),
        completed_cells=("cells", "sum")).reset_index()
    endpoints["metric"] = endpoints.task.map(METRICS)
    endpoints["rank"] = np.nan
    endpoints["normalized_regret"] = np.nan
    for _, group in endpoints.groupby(["task", "endpoint", "regime"]):
        values = group.mean_score
        lower_better = group.task.iloc[0] == "barrier"
        ranks = values.rank(ascending=lower_better, method="average")
        width = float(values.max() - values.min())
        regrets = ((values - values.min()) if lower_better else (values.max() - values)) / width if width > 1e-12 else values * 0
        endpoints.loc[group.index, "rank"] = ranks
        endpoints.loc[group.index, "normalized_regret"] = regrets
    families = endpoints.groupby(["task", "family", "regime", "config"], dropna=False).agg(
        mean_rank=("rank", "mean"), mean_regret=("normalized_regret", "mean"),
        endpoints=("endpoint", "nunique"), encoding_coverage=("encoding_coverage", "mean")).reset_index()
    task_regimes = families.groupby(["task", "regime", "config"], dropna=False).agg(
        mean_family_rank=("mean_rank", "mean"), mean_family_regret=("mean_regret", "mean"),
        source_families=("family", "nunique"), encoding_coverage=("encoding_coverage", "mean")).reset_index()
    candidate_frames = []
    task_views = []
    for view, requested in (("grouped_main", "grouped_cv"), ("native_reference", "native")):
        chosen = families[(~families.task.isin(["reagent", "condition"])) | families.regime.eq(requested)]
        for task in ("reagent", "condition"):
            if chosen[chosen.task.eq(task)].empty:
                raise ValueError("Missing separate native/grouped representative view")
        # Regimes inside a source family are balanced first. Neither the number
        # of optimizer runs nor duplicate hierarchical endpoints increase weight.
        family_views = chosen.groupby(["task", "family", "config"], dropna=False).agg(
            rank=("mean_rank", "mean"), regret=("mean_regret", "mean"),
            encoding_coverage=("encoding_coverage", "mean"), regimes=("regime", "nunique")).reset_index()
        task_view = family_views.groupby(["task", "config"]).agg(
            rank=("rank", "mean"), regret=("regret", "mean"),
            encoding_coverage=("encoding_coverage", "mean"), source_families=("family", "nunique")).reset_index()
        if set(task_view.task) != set(TASKS):
            raise ValueError("A macro view cannot omit a task")
        task_view["view"] = view
        task_views.append(task_view)
        candidates = task_view.groupby("config").agg(macro_rank=("rank", "mean"),
            macro_regret=("regret", "mean"), worst_task_rank=("rank", "max"),
            encoding_coverage=("encoding_coverage", "mean"), represented_tasks=("task", "nunique")).reset_index()
        if not candidates.represented_tasks.eq(7).all():
            raise ValueError("Macro representative lacks complete task coverage")
        for task in TASKS:
            values = task_view[task_view.task.eq(task)].set_index("config")
            candidates[task + "_rank"] = candidates.config.map(values["rank"])
            candidates[task + "_regret"] = candidates.config.map(values["regret"])
        candidates["view"] = view
        candidates["interpretation"] = "exploratory; no configuration selected"
        candidate_frames.append(candidates)
    return dict(fold_seed_summary=folds, endpoint_summary=endpoints, family_summary=families,
                task_regime_summary=task_regimes, task_view_summary=pd.concat(task_views, ignore_index=True),
                common_candidates=pd.concat(candidate_frames, ignore_index=True))


def report(root=S.ROOT, old_root=OLD_ROOT, old_export=OLD_EXPORT, new_export=S.EXPORT, output=None):
    new_root = Path(root)
    old_root, old_export, new_root, new_export = map(Path, (old_root, old_export, new_root, new_export))
    output = Path(output) if output else new_root / "common_representative_analysis"
    completed_path = new_root / "COMMON_TASK_ANALYSIS_COMPLETE.json"
    if completed_path.exists():
        previous = read(completed_path)
        if (previous.get("phase") != "complete" or previous.get("configuration_selected") is not None
                or previous.get("source_sha256") != S.sha(__file__)
                or previous.get("output_root") != str(output.resolve())):
            raise ValueError("Changed or unrecognized completed common analysis")
        for identity in previous["sources"]:
            for path_key, digest_key in (("csv", "csv_sha256"), ("complete", "complete_sha256"), ("plan", "plan_sha256")):
                if S.sha(identity[path_key]) != identity[digest_key]:
                    raise ValueError("Completed common-analysis input changed")
        for name, digest in previous["outputs_sha256"].items():
            if S.sha(output/name) != digest:
                raise ValueError("Completed common-analysis output changed")
        return previous
    provenance, normalized = [], []
    old_plan = read(old_root / "plan.json")
    new_plan = S.verify(new_root)
    if old_plan["configs"] != S.CONFIGS:
        raise ValueError("The completed study22 and study30 construction grids differ")
    for task, expected in OLD_COUNTS.items():
        frame, identity = canonical_cells(old_export/task/"cells.csv", old_export/task/"COMPLETE.json",
                                           old_root/"plan.json", expected, task)
        normalized.append(old_normalized(frame, task)); provenance.append(identity)
    frame, identity = canonical_cells(new_export/"cells.csv", new_root/"COMPLETE.json", new_root/"plan.json",
                                       new_plan["expected_models"], new=True)
    if "n_test" not in frame and "test_rows" not in frame:
        # Exact stored native selectors supply counts when the metric CSV did
        # not repeat them. No independent sample count is inferred from fits.
        split_counts = {(source, split["name"]): split.get("test_rows", split.get("expert_rows"))
                        for source, info in new_plan["sources"].items() for split in info["splits"]}
        frame["test_rows"] = [split_counts[(str(row.source),str(row.split))] for row in frame.itertuples()]
        if frame.test_rows.isna().any():
            raise ValueError("Native selectors do not provide exact held-out row counts")
    normalized.append(new_normalized(frame)); provenance.append(identity)
    cells = pd.concat(normalized, ignore_index=True)
    outputs = analyze(cells)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        value.to_csv(output/(name+".csv"), index=False)
    lines = ["# Common SynRFP representative diagnostics", "",
        "All seven task factorials are complete. This table supports choosing one common preset; no preset is selected automatically.", "",
        "Scores are exploratory because historical test outcomes were viewed. Ranks and regrets are balanced within source families and then across seven tasks.", "",
        "The grouped_main view uses grouped reagent/condition evaluations. The native_reference view substitutes their author splits; those regimes are never averaged together.", "",
        "Yield retains all four original regimes with separate rows; random and OOD regimes inside each source family receive equal weight. Barrier families link GDB7/RDB7 and E2/SN2/E2SN2.", "",
        "Study22 clustering fixes K from the endpoint row count using min(512,max(2,round(sqrt(n/2)))); class labels and class counts do not choose K. These factorial cells remain separate from the main known-class-count comparison and adaptive unknown-K clustering experiments.", "",
        "Condition fold means average five optimizer seeds first. Native/FREA repeated seeds use one fixed test set. Fold training populations and repeated holdouts overlap; repeated_test_rows_sum is not an independent sample size.", "",
        "Encoding coverage is descriptive:study22 classification/yield use held-out coverage, study22 clustering uses all input rows, and study30 uses the frozen task population. Its macro is a mean reported fraction, not a new pooled coverage estimate.", "",
        "Cost:the common candidate table does not invent missing per-fit durations. Study30's cell table currently omits those columns; source-specific encoding and training receipts retain cost evidence for a separate measured cost analysis.", "",
        "NCS BH includes some source material related to classic BH HTE; source-family balance is a transparent weighting convention, not a claim of dataset independence. EHReact is excluded.", "",
        "Read common_candidates.csv together with endpoint_summary.csv, family_summary.csv and task_regime_summary.csv. Different metrics or regression units are never directly pooled.", ""]
    (output/"report.md").write_text("\n".join(lines))
    for identity in provenance:
        for path_key, digest_key in (("csv", "csv_sha256"), ("complete", "complete_sha256"), ("plan", "plan_sha256")):
            if S.sha(identity[path_key]) != identity[digest_key]:
                raise ValueError("Source report changed during common analysis")
    files = [name+".csv" for name in outputs] + ["report.md"]
    receipt = dict(schema="synrfp-seven-task-common-candidate-report-v1", phase="complete", tasks=list(TASKS),
                   configurations=len(S.CONFIGS), logical_input_cells=len(cells), sources=provenance,
                   source_sha256=S.sha(__file__), interpretation="exploratory_after_prior_test_results_viewed",
                   configuration_selected=None, metric_by_task=METRICS, views=["grouped_main", "native_reference"],
                   aggregation="fold/seed -> endpoint/regime -> source family -> equal task rank/regret",
                   test_sets_independent=False, outputs_sha256={name:S.sha(output/name) for name in files},
                   output_root=str(output.resolve()), cost_scope="Incomplete cross-task cell-level cost columns; consult native fit/encoding receipts",
                   finished_utc=S.now())
    S.write(output/"COMPLETE.json", receipt)
    S.write(completed_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--old-export", type=Path, default=OLD_EXPORT)
    parser.add_argument("--new-root", type=Path, default=S.ROOT)
    parser.add_argument("--new-export", type=Path, default=S.EXPORT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(root=args.new_root,old_root=args.old_root,old_export=args.old_export,
                            new_export=args.new_export,output=args.output),indent=2))
