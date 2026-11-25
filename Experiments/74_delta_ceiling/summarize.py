"""Produce label-purity reports from the exact count-class keys and frozen splits."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from encode_counts import CHANNELS, check, digest, read, write

CONTEXT = {}


def majority(group, label, n_group, n_label):
    pair, count = np.unique(group.astype(np.int64)*n_label+label, return_counts=True)
    majority = np.zeros(n_group, dtype=np.int64)
    np.maximum.at(majority, pair//n_label, count)
    n_labels = np.bincount(pair//n_label, minlength=n_group)
    return majority, n_labels


def groups_for(keys):
    _, indexes = np.unique(keys, return_inverse=True)
    return indexes.astype(np.int32)


def same_partition(first, second):
    n = int(first.max())+1
    lookup = np.empty(n, dtype=np.int32)
    lookup[first] = second
    return int(second.max())+1 == n and np.array_equal(lookup[first], second)


def published_selectors(endpoint, panel, base, source_positions):
    plan = Path(base)/"default_optimal_benchmark_v1"
    selectors, integrity = [], {}
    for seed in range(25):
        code_path = Path(panel["split_paths"][str(seed)])
        fit_hash = check(code_path, panel["split_sha256"][str(seed)])
        code = np.load(code_path, allow_pickle=False)
        np.testing.assert_array_equal(np.unique(code), [0,1,2])
        fit = plan/"results"/f"classification__{endpoint}__mlp__{seed}__synrfp_optimal"/"result.json"
        receipt = read(fit)
        if receipt["dataset"] != endpoint or receipt["seed"] != seed or receipt["identity"]["split_sha256"] != fit_hash:
            raise ValueError(f"Changed fitting identity {endpoint}/{seed}")
        artifacts = Path(receipt["attempt_directory"])
        pin = receipt["artifacts_sha256"]
        state = artifacts/"state.npz"
        row_file = artifacts/"selection.npz"
        check(state, pin["state.npz"])
        check(row_file, pin["selection.npz"])
        with np.load(row_file, allow_pickle=False) as selected:
            outer, test = selected["outer"], selected["test"]
        panel_outer = np.r_[np.flatnonzero(code==0),np.flatnonzero(code==1)]
        panel_test = np.flatnonzero(code==2)
        np.testing.assert_array_equal(outer, source_positions[panel_outer])
        np.testing.assert_array_equal(test, source_positions[panel_test])
        with np.load(state, allow_pickle=False) as checkpoint:
            actual, valid = checkpoint["train"], checkpoint["val"]
        np.testing.assert_array_equal(np.sort(np.r_[actual,valid]), np.arange(len(outer)))
        selectors.append((panel_outer[actual],panel_test))
        integrity[str(seed)] = {"split_sha256":fit_hash,"optimizer_selection_state_sha256":pin["state.npz"],
                                "result_sha256":digest(fit),"selection_sha256":pin["selection.npz"]}
    for repeat in range(5):
        np.testing.assert_array_equal(np.sort(np.concatenate([test for _,test in selectors[5*repeat:5*repeat+5]])), np.arange(panel["n"]))
    return selectors, integrity


def examples(endpoint, groups, labels, mixed, heavy, ids, reactions, raw, source_positions):
    # The two smallest distinct-label representatives form the smallest pair
    # in each class. Rank classes by pair size, then stable source IDs.
    candidates = np.flatnonzero(mixed[groups])
    ordered = candidates[np.lexsort((ids[candidates], heavy[candidates], groups[candidates]))]
    pairs = {}
    for position in ordered:
        group = int(groups[position])
        if group not in pairs:
            pairs[group] = [position]
        elif len(pairs[group]) == 1 and labels[position] != labels[pairs[group][0]]:
            pairs[group].append(position)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("A mixed group did not yield a label-disagreeing pair")
    selected = sorted(pairs.values(),key=lambda pair:(int(heavy[pair].sum()),*ids[pair]))[:20]
    output = []
    for i, (left, right) in enumerate(selected, start=1):
        item = {"endpoint":endpoint,"pair":i,"group_id":int(groups[left]),
                "total_heavy_atoms":int(heavy[left]+heavy[right])}
        for suffix, index in (("a",left),("b",right)):
            source = raw.iloc[source_positions[index]]
            item.update({f"r_id_{suffix}":ids[index],f"rxn_{suffix}":reactions[index],
                         f"heavy_atoms_{suffix}":int(heavy[index]),
                         f"source_position_{suffix}":int(source_positions[index])})
            item.update({f"ec{level}_{suffix}":source[f"ec{level}"] for level in (1,2,3)})
        label = int(endpoint[-1])
        if item[f"ec{label}_a"] == item[f"ec{label}_b"]:
            raise ValueError("Raw EC labels disagree with the benchmark coding")
        output.append(item)
    return output


def analyze(job):
    endpoint, panel = job
    base, local = CONTEXT["base"], CONTEXT["local"]
    name = panel["source"]
    source_file = Path(CONTEXT["sources"][name]["table"])
    source_proof = CONTEXT["encoded"]["sources"][name]
    check(source_file, source_proof["source_sha256"])
    check(panel["table"],panel["table_sha256"])
    source = pd.read_csv(source_file,dtype={"row_id":str})
    population = pd.read_csv(panel["table"],dtype={"row_id":str})
    np.testing.assert_array_equal(population.dataset_position,np.arange(panel["n"]))
    np.testing.assert_array_equal(population.row_id,source.row_id.to_numpy()[population.source_position])
    archive = local/(name+".npz")
    check(archive,source_proof["artifact_sha256"])
    with np.load(archive,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["row_ids"],source.row_id.to_numpy(str))
        pos = population.source_position.to_numpy()
        ok = saved["ok"][pos]
        keys = saved["keys"][pos[ok]]
        heavy = saved["heavy"][pos[ok]]
    source_positions = pos[ok]
    n = len(source_positions)
    ids = source.row_id.to_numpy(str)[source_positions]
    eligible = np.full(panel["n"],-1,dtype=np.int32)
    eligible[ok] = np.arange(n,dtype=np.int32)
    labels = np.unique(population.y.to_numpy()[ok],return_inverse=True)[1]
    n_labels = int(labels.max())+1
    selectors, integrity = published_selectors(endpoint,panel,base,pos)
    fit_test = [(eligible[fit][ok[fit]],eligible[test][ok[test]]) for fit,test in selectors]
    fold_total = sum(len(test) for _,test in fit_test)
    assert fold_total == 5*n
    endings, per_fold, example_rows = [], [], []
    shard = local/f"{endpoint}.groups.csv.gz"
    stats = local/f"{endpoint}.summary.json"
    with gzip.open(shard,"wt",newline="",compresslevel=3) as groups_out:
        for p, profile in enumerate(CONTEXT["profiles"]):
            groupings = [groups_for(keys[:,p,c]) for c in range(4)]
            if not same_partition(groupings[2],groupings[3]):
                raise ValueError("Exact paired channels violated U=|D|+2S")
            for channel, groups in zip(CHANNELS,groupings,strict=True):
                count = np.bincount(groups)
                n_group = len(count)
                top, n_in_group = majority(groups,labels,n_group,n_labels)
                mixed = n_in_group>1
                seen_n, seen_top = 0,0
                for seed,(fit,test) in enumerate(fit_test):
                    learned = np.zeros(n_group,dtype=bool)
                    learned[groups[fit]] = True
                    included = test[learned[groups[test]]]
                    top_held,_ = majority(groups[included],labels[included],n_group,n_labels)
                    seen, correct = len(included),int(top_held.sum())
                    per_fold.append({"endpoint":endpoint,"tokenizer":profile,"channel_set":channel,
                                     "repeat":seed//5,"fold":seed%5,
                                     "eligible_test_rows":len(test),"seen_test_rows":seen,
                                     "seen_test_majority_count":correct,
                                     "seen_ceiling":correct/seen if seen else None})
                    seen_n += seen
                    seen_top += correct
                endings.append({"endpoint":endpoint,"tokenizer":profile,"channel_set":channel,
                                "n_reactions":n,"n_groups":n_group,"n_singleton_groups":int((count==1).sum()),
                                "collision_share":float(count[mixed].sum()/n),
                                "ceiling":float(top.sum()/n),
                                "study65_rows":panel["n"],"encoding_refusals":int((~ok).sum()),
                                "seen_test_events":seen_n,"encoded_test_events":fold_total,
                                "seen_test_fraction":seen_n/fold_total,
                                "seen_ceiling":seen_top/seen_n if seen_n else None})
                groups_out.writelines(
                    f"{endpoint},{profile},{channel},{i},{size},{labels_here},{share:.10g}\n"
                    for i,(size,labels_here,share) in enumerate(zip(count,n_in_group,top/count,strict=True))
                )
                if endpoint.startswith("ecreact") and profile=="wl_r2" and channel=="D":
                    raw = pd.read_csv(CONTEXT["ec_raw"],dtype=str)
                    np.testing.assert_array_equal(raw.r_id,source.row_id)
                    ec = raw.iloc[source_positions][f"ec{endpoint[-1]}"].to_numpy(str)
                    if len(np.unique(ec,return_inverse=True)[0]) != n_labels:
                        raise ValueError("EC label count differs from published endpoint")
                    pairs = set(zip(ec,labels,strict=True))
                    if len(pairs)!=n_labels:
                        raise ValueError("Raw EC labels are not bijective with endpoint codes")
                    example_rows = examples(endpoint,groups,labels,mixed,heavy,ids,
                                            source.rxn.to_numpy(str)[source_positions],raw,source_positions)
    payload = {"endpoint":endpoint,"source":name,"summary":endings,"folds":per_fold,"examples":example_rows,
               "study65_rows":panel["n"],"encoded":n,"dropped":panel["n"]-n,
               "panel_sha256":panel["table_sha256"],"selector_sha256":integrity,
               "group_shard_sha256":digest(shard)}
    write(stats,payload)
    print(f"{endpoint}: {n:,} valid, {panel['n']-n} encoding refusals",flush=True)
    return stats.name


def export_csv(path, rows):
    with path.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(folder, summary, contrasts):
    table = {(r["endpoint"],r["channel_set"]):r for r in summary if r["tokenizer"]=="wl_r2"}
    lines = [
        "# Exact count ceilings for classification",
        "",
        "Study 65's native cleaned classification panels supply the rows and labels.",
        "The analysis uses full occurrence counts before any sketching. The frozen",
        "encoder refusals are excluded from the ceilings and retain their original",
        "classification benchmark status. Four tokenizer/radius profiles are reported.",
        "",
        "| Endpoint | Encoded / study 65 | D mixed-label row share | D oracle accuracy | D+U oracle accuracy | D+U minus D oracle (pp) | Published MLP MCC change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for endpoint in sorted({r["endpoint"] for r in summary}):
        d, du = table[endpoint,"D"], table[endpoint,"D+U"]
        lines.append(f"| {endpoint} | {d['n_reactions']:,} / {d['study65_rows']:,} | {d['collision_share']:.3%} | "
                     f"{d['ceiling']:.3%} | {du['ceiling']:.3%} | {100*(du['ceiling']-d['ceiling']):+.3f} | {contrasts[endpoint]:+.3f} |")
    lines.extend([
        "", "The MCC differences average the 18 tokenizer, radius and sketch settings",
        "in the pinned paper source table. The oracle values above use WL radius 2.",
        "These are two different quantities. A near-perfect full-cohort oracle can",
        "arise from singleton groups and offers little information about prediction",
        "on an exact class absent from fitting.",
        "", "For each of the 25 saved grouped partitions, the seen-class ceiling uses",
        "only test rows whose exact class appears among the retained MLP optimizer",
        "fitting rows. It sums the majorities of their held-out label histograms.",
        "The summary pools eligible test events over the 25 partitions and reports",
        "their fraction of all encoded test events. The fold file gives denominators",
        "and ceilings separately. These values use held-out labels to define an oracle.",
        "", "D+U and D+S induce identical exact partitions because U = |D| + 2S.",
        "The implementation independently checks that equivalence for every endpoint",
        "and tokenizer. Their finite-width sketches can behave differently.",
        "", "Group IDs are ordinals scoped to endpoint, tokenizer and channel set.",
        "The ECReact examples use the smallest distinct-label pair per exact D class",
        "and rank those pairs by total heavy atoms on both reaction sides.",
        "", "See [endpoint_summary.csv](endpoint_summary.csv), [fold_summary.csv](fold_summary.csv),",
        "[group_table.csv.gz](group_table.csv.gz), [examples.csv](examples.csv), and [AUDIT.json](AUDIT.json).",
    ])
    (folder/"README.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base","encoding","ec-raw","context-mcc","output"):
        parser.add_argument(f"--{field}",type=Path,required=True)
    parser.add_argument("--jobs",type=int,default=8)
    args = parser.parse_args()
    local = args.encoding
    encoded = read(local/"ENCODED.json")
    check(local/"ENCODING_PLAN.json",encoded["plan_sha256"])
    plan = read(local/"ENCODING_PLAN.json")
    base = args.base
    class_plan = base/"default_optimal_benchmark_v1/native/classification/plan.json"
    old_plan = base/"all_endpoint_factorial_full_v1/plan.json"
    for path in (class_plan,old_plan):
        check(path,plan["input_plans_sha256"][str(path)])
    pins = read(class_plan)
    CONTEXT.update(base=base,local=local,encoded=encoded,sources=read(old_plan)["sources"],
                   profiles=plan["token_profiles"],ec_raw=args.ec_raw)
    ec_raw_sha = "c730e5bd196dc69b5ebc421cce37010162f7e129f149b28196f0069ee242bdcd"
    check(args.ec_raw,ec_raw_sha)
    changes = pd.read_csv(args.context_mcc)
    classification = changes.loc[changes.task=="classification"]
    averages = classification.groupby("dataset").benefit_difference.agg(["size","mean"])
    if set(averages.index)!=set(pins["panels"]) or not (averages["size"]==18).all():
        raise ValueError("Expected 18 published context contrasts for every endpoint")
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        names = list(executor.map(analyze,sorted(pins["panels"].items())))
    items = [read(local/name) for name in names]
    summary = [row for item in items for row in item["summary"]]
    for row in summary:
        row["published_mlp_context_mcc_change"] = float(averages.loc[row["endpoint"],"mean"])
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    export_csv(out/"endpoint_summary.csv",summary)
    export_csv(out/"fold_summary.csv",[row for item in items for row in item["folds"]])
    export_csv(out/"examples.csv",[row for item in items for row in item["examples"]])
    with (out/"group_table.csv.gz").open("wb") as handle:
        handle.write(gzip.compress(b"endpoint,tokenizer,channel_set,group_id,size,n_labels,majority_share\n",mtime=0))
        for item in items:
            source = local/f"{item['endpoint']}.groups.csv.gz"
            check(source,item["group_shard_sha256"])
            with source.open("rb") as part:
                shutil.copyfileobj(part,handle)
    markdown(out,summary,averages["mean"].to_dict())
    audit = {
        "status":"complete","model_fits":0,"encoding":plan,
        "summarizer_sha256":digest(Path(__file__)),"cohort_source_receipts":encoded["sources"],
        "ec_raw_sha256":ec_raw_sha,"paper_context_changes_csv_sha256":digest(args.context_mcc),
        "paper_context_revision":"ac83ccf",
        "endpoints":{item["endpoint"]:{k:v for k,v in item.items()
                     if k not in ("endpoint","summary","folds","examples","group_shard_sha256")} for item in items},
        "classes_use": "SHA256 of sorted native token counts; no fingerprint sketch",
        "fit_seen_definition": "Test events whose exact key appears in that fold's fitted optimizer rows",
        "no_sketch":True,"d_u_d_s_partition_equality_verified":True,
    }
    write(out/"AUDIT.json",audit)
    outputs = {file.name:digest(file) for file in out.iterdir()
               if file.is_file() and file.name!="COMPLETE.json"}
    write(out/"COMPLETE.json",{"status":"complete","endpoints":len(items),
                              "profiles":plan["token_profiles"],"summary_cells":len(summary),
                              "fold_cells":sum(len(item["folds"]) for item in items),
                              "examples":sum(len(item["examples"]) for item in items),
                              "files_sha256":outputs})
    print(f"Published {len(summary)} endpoint/profile/channel summaries",flush=True)


if __name__ == "__main__":
    main()
