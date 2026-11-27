"""Audit the complete crossing and export label-free operating points."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from statistics import mean, stdev
import math
import platform

import pandas as pd

from prepare91 import HERE, ORIGINAL, ROOT, R79, SEEDS, STANDARD, THRESHOLDS, read, sha, write
from run91 import identity


METRICS = ("ami","ari","nmi","n_clusters","singleton_share","largest_cluster_share",
           "runtime_seconds","fit_and_score_seconds")
OUTPUT = HERE / "publication_metadata/bitbirch_factorial_v1"


def members(plan, plan_sha):
    for source_name, source in plan["sources"].items():
        for group_name, group in source["groups"].items():
            for name, input_info in source["feature_files"].items():
                for order in SEEDS:
                    p = identity(source_name, group_name, name, order)
                    row = read(p)
                    assert row["plan_sha256"] == plan_sha
                    assert row["feature_sha256"] == input_info["sha256"]
                    assert row["order_seed"] == order and row["representation"] == name
                    assert row["source"] == source_name and row["population"] == group_name
                    conversion = input_info["conversion"]
                    assert row["conversion"] == conversion
                    scores = row["scores"]
                    assert set(scores) == {str(t) for t in THRESHOLDS}
                    for t in THRESHOLDS:
                        stats = scores[str(t)]
                        assert stats["n_rows"] == group["n"]
                        assert {s["endpoint"] for s in stats["endpoints"]} == set(group["endpoints"])
                        for values in stats["endpoints"]:
                            yield dict(representation=name, binarized=conversion!="native", conversion=conversion,
                                       population=group_name, source=source_name, threshold=t / 100,
                                       order_seed=order, n_rows=stats["n_rows"],
                                       n_refusal_placeholders=stats["n_refusal_placeholders"],
                                       runtime_seconds=stats["fit_seconds"],
                                       fit_and_score_seconds=stats["cpu_batch_seconds"],
                                       ami_arithmetic=stats.get("ami_arithmetic", "sklearn_native"),
                                       partition_sha256=stats["assignment_sha256"], **values)


def analyze(cells, plan):
    cells = cells.sort_values(["endpoint","representation","threshold","order_seed"], kind="stable")
    assert len(cells) == plan["expected_scores"]
    assert not cells.duplicated(["endpoint","representation","threshold","order_seed"]).any()
    grouping = ["endpoint","representation","threshold","binarized","conversion","n_rows",
                "n_refusal_placeholders"]
    aggregated = []
    for key, subset in cells.groupby(grouping):
        if tuple(subset.order_seed) != SEEDS:
            raise ValueError(f"Incomplete five-order crossing for {key}")
        record = dict(zip(grouping, key, strict=True))
        for metric in METRICS:
            record[f"{metric}_mean"] = mean(subset[metric])
            record[f"{metric}_sample_sd"] = stdev(subset[metric].astype(float))
        aggregated.append(record | {"n_orders": len(subset)})
    summary = pd.DataFrame(aggregated)
    settings = []
    for _, group in summary.groupby(["endpoint","representation"]):
        assert tuple(group.threshold) == tuple(t/100 for t in THRESHOLDS)
        n = group.iloc[0].n_rows
        k = min(512,max(2,round(math.sqrt(n/2))))
        pos = min(group.index,key=lambda i: (abs(group.at[i,"n_clusters_mean"]-k),
                                             group.at[i,"threshold"]))
        for name, i in (("fixed_0.35",group.index[group.threshold.eq(.35)][0]),
                        ("closest_to_row_count_k",pos)):
            chosen = group.loc[i].to_dict()
            settings.append(chosen | dict(operating_point=name, row_count_rule_k=k,
                difference_from_k=chosen["n_clusters_mean"]-k))
    return cells, summary, pd.DataFrame(settings)


def decomposition(cells, points):
    native_path = ORIGINAL / "Experiments/47_morgan_all_tasks/factor_analysis.py"
    spec = spec_from_file_location("factor_analysis91", native_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.METRICS = {**module.METRICS, "clustering":"ami"}
    c = cells.merge(points.loc[(points.operating_point=="closest_to_row_count_k") &
        points.representation.str.startswith("synrfp-"),["representation","endpoint","threshold"]],
        on=["representation","endpoint","threshold"],validate="many_to_one")
    records = [dict(task="clustering",dataset=r.endpoint,protocol="bitbirch_closest_to_row_count_k",
                    config=r.representation,split=str(r.order_seed),ami=r.ami) for r in c.itertuples()]
    result = module.analyze(records)["interaction_variance.csv"]
    assert len(result) == 12 * 15
    return pd.DataFrame(result)


def attach_kmeans(points):
    roots = ORIGINAL / "Experiments"
    specs = (
        (roots/"56_factorial_refresh/publication_metadata/corrected_full_cohort_v1", "cells.csv"),
        (roots/"65_default_optimal/publication_metadata/default_optimal_v1", "clustering_blind/cells.csv"),
    )
    frames, hashes = [], {}
    for root, relative in specs:
        complete, table = root/"COMPLETE.json", root/relative
        expected = read(complete)["exported_outputs_sha256"][relative]
        assert sha(table) == expected
        hashes[str(table)] = expected
        hashes[str(complete)] = sha(complete)
        rows = pd.read_csv(table, low_memory=False)
        if relative == "cells.csv":
            rows = rows.query("task == 'clustering' and regime == 'heuristic_rowcount_kmeans'")
            rows = rows.rename(columns={"config":"representation","test_rows":"n_rows","score":"ami"})
        else:
            rows = rows.query("method in ('drfp','rdkit_struct')")
            rows = rows.rename(columns={"dataset":"endpoint","method":"representation","n":"n_rows"})
        frames.append(rows[["endpoint","representation","n_rows","ami"]])
    rows = pd.concat(frames, ignore_index=True)
    table = rows.groupby(["endpoint","representation","n_rows"],as_index=False).ami.agg(
        row_count_kmeans_ami_mean="mean",row_count_kmeans_ami_sample_sd="std",row_count_kmeans_runs="size")
    assert len(table) == 12 * 74
    output = points.merge(table, on=["endpoint","representation","n_rows"],how="left",validate="many_to_one")
    assert output.row_count_kmeans_runs.notna().sum() == 12 * 74 * 2
    return output, hashes


def replay(plan, cells):
    matched = cells.loc[(cells.representation == STANDARD) & cells.threshold.eq(.35)]
    max_error = 0.
    for r in matched.itertuples():
        reference = read(R79/r.source/f"{r.population}__d2048__{r.order_seed}.json")
        exact = next(s for s in reference["scores"] if s["endpoint"] == r.endpoint)
        assert exact["n_rows"] == r.n_rows and exact["n_clusters"] == r.n_clusters
        assert exact["n_refusal_placeholders"] == r.n_refusal_placeholders
        max_error=max(max_error,abs(exact["ami"]-r.ami),abs(exact["ari"]-r.ari),
                      abs(exact["singleton_share"]-r.singleton_share))
    assert len(matched) == plan["n_endpoints"] * len(SEEDS)
    assert max_error <= 1e-8
    return {"matched_request79":len(matched),"max_absolute_error":max_error}


def report_note(plan, fit, summary, operating, check, cpu):
    skipped = "\n".join(f"- {name}. {why}" for name,why in plan["skips"].items()) or "- None."
    rows = operating.loc[operating.operating_point.eq("closest_to_row_count_k")]
    within_10 = float((rows.difference_from_k.abs() <= 0.10*rows.row_count_rule_k).mean())
    return f"""# Full BitBIRCH factorial and benchmark

The study completed {plan['expected_fits']:,} CPU fits across ten full populations, with
{plan['expected_scores']:,} endpoint scores. ECReact levels share each clustering fit,
with separate scores for their reference labels. The three repaired TPL B rows use the
updated released caches. All other frozen cohort rows, including refusal placeholders,
are included. Morgan uses the corrected graph-native profile.

The sweep uses Tanimoto, branching factor 50, diameter merging, thresholds 0.20 to
0.80 and matched input-order seeds 20260924–20260928. The first operating point uses
threshold 0.35. The second minimizes the absolute difference between mean occupied
clusters and the k-means row-count rule, with ties resolved by the lower threshold.
In {within_10:.1%} of the reported second operating points, the occupied count is within
10% of that rule. The full signed differences are in `operating_points.csv`.
No class labels enter threshold selection. The operating-point table also includes
the released row-count k-means AMI on the same rows for the factorial and the two
native-bit baselines.

The baseline rows record their conversion explicitly. DRFP and RDKit structural use
native bits. RDKit difference maps nonzero counts to one. RXNFP-MLM uses positive
coordinates. The binarized results describe those projections. Singleton share is the
fraction of observations that are alone in a cluster. Standard deviations describe
five input orders of each fixed cohort.

## Skipped representation

{skipped}

## Runtime and reproducibility

The workstation uses a {cpu}. The main service used {fit['jobs']} CPU workers and took
{fit['elapsed_seconds']/3600:.2f} wall-clock hours. The initial pilot results are reused
and timed in the cells. The full report contains {len(summary):,} setting summaries.
The {check['matched_request79']} overlapping Request 79 cells were replayed, with a
maximum absolute score difference of {check['max_absolute_error']:.2g}. The variance
table uses the same balanced decomposition as the k-means factorial and includes all
interaction orders. The release manifest pins cache files, source code and software versions.
The grouped AMI calculation sums the same hypergeometric terms by cluster-size
multiplicity. Full-cohort controls verify agreement with the native sklearn metric.
"""


def main():
    manifest = ROOT/"PREPARED.json"
    plan = read(manifest)
    plan_sha = sha(manifest)
    fit = read(ROOT/"FIT_COMPLETE.json")
    assert not fit["failures"] and fit["plan_sha256"] == plan_sha
    cells, summary, operating = analyze(pd.DataFrame(members(plan, plan_sha)), plan)
    variance = decomposition(cells, operating)
    operating, kmeans_hashes = attach_kmeans(operating)
    check = replay(plan, cells)
    cpu = next((x.split(":",1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                if x.startswith("model name")),platform.processor())
    OUTPUT.mkdir(parents=True,exist_ok=True)
    tables = {"cells.csv":cells, "summary.csv":summary, "operating_points.csv":operating,
              "factorial_variance.csv":variance}
    for name, table in tables.items():
        table.to_csv(OUTPUT/name,index=False)
    (OUTPUT/"note.md").write_text(report_note(plan, fit, summary, operating, check, cpu))
    write(OUTPUT/"COMPLETE.json",dict(status="complete", prepare_sha256=plan_sha,
        prepare=plan, execution=fit, replay=check, grouped_ami_validation=read(ROOT/"GROUPED_AMI_VERIFIED.json"),
        matching_kmeans_sha256=kmeans_hashes,
        outputs_sha256={name:sha(OUTPUT/name) for name in (*tables,"note.md")},
        code_sha256={p.name:sha(p) for p in HERE.glob("*.py")}))
    print("Closed",plan["expected_fits"],"fits",len(cells),"scores",flush=True)


if __name__ == "__main__":
    main()
