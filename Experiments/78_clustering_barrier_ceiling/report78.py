"""Independently validate Request 78 outputs and prepare the publication bundle."""
from itertools import product
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from common78 import HERE, LOCAL, OUTPUT, digest, paper_table, Pins, read, write

SETTINGS = ("d2048", "u0", "u0.25", "u0.5", "u0.75", "u1")
PROTOCOLS = ("known_k", "factorial_k")


def check_files(root, manifest):
    for name, expected in manifest.items():
        assert digest(root / name) == expected, name


def check_clustering(root):
    audit = read(root / "clustering_AUDIT.json")
    check_files(root, audit["outputs_sha256"])
    for name, expected in audit["input_sha256"].items():
        if name.startswith("git:"):
            continue
        assert digest(Path(name)) == expected, name
    paper = paper_table(Pins())
    arch = paper.query("task == 'clustering'").groupby("dataset").benefit_difference.mean()
    summaries = pd.read_csv(root / "clustering_weight_scan.csv")
    raw = pd.read_csv(root / "clustering_weight_cells.csv")
    pairs = pd.read_csv(root / "clustering_distance_ratio.csv")
    endpoints = sorted(arch.index)
    cells = set(zip(raw.endpoint, raw.protocol, raw.setting, raw.seed, strict=True))
    assert cells == set(product(endpoints, PROTOCOLS, SETTINGS, range(5)))
    assert len(raw) == len(cells) == 720
    assert raw.origin.eq("new_weight_scan").all() and audit["new_fits"] == 720
    tpl = raw.endpoint.str.startswith("tpl_")
    assert raw.loc[tpl, "input_layout"].eq("float32_dense_packed_init").all()
    assert raw.loc[~tpl, "input_layout"].eq("float32_csr").all()
    anchored = raw.loc[raw.published_factorial_ami.notna()].copy()
    assert len(anchored) == audit["archived_factorial_scores"] == 120
    assert anchored.protocol.eq("factorial_k").all()
    assert set(anchored.setting) == {"d2048", "u1"}
    np.testing.assert_allclose(anchored.ami-anchored.published_factorial_ami,
                               anchored.ami_minus_published_factorial, rtol=0, atol=1e-12)
    delta = anchored.ami_minus_published_factorial.abs()
    checks = {"archive_anchors":len(anchored),"match_at_1e_12":int(delta.le(1e-12).sum()),
              "max_abs_ami_delta":float(delta.max()),"median_abs_ami_delta":float(delta.median())}
    assert set(zip(pairs.endpoint, pairs.block, strict=True)) == set(product(endpoints, ("D", "U")))
    assert pairs.shape[0] == 24
    assert pairs.n_within_pairs.le(2000).all() and pairs.n_between_pairs.le(2000).all()
    np.testing.assert_allclose(pairs.between_mean_squared_euclidean / pairs.within_mean_squared_euclidean,
                               pairs.between_over_within, rtol=0, atol=1e-12)
    indexes = ["endpoint", "protocol", "setting"]
    replay = raw.groupby(indexes).ami.agg(["mean", "std", "size"]).sort_index()
    supplied = summaries.set_index(indexes).sort_index()
    assert supplied.index.equals(replay.index)
    for actual, expected in (("mean", "ami_mean"), ("std", "ami_sd"), ("size", "n_seeds")):
        np.testing.assert_allclose(replay[actual], supplied[expected], rtol=0, atol=1e-12)
    for endpoint in endpoints:
        selected = raw.query("endpoint == @endpoint")
        assert selected.n_rows.nunique() == 1
        assert selected.n_refusal_placeholders.nunique() == 1
        np.testing.assert_allclose(selected.paper_mean_context_ami_change_over_18,
                                   arch.loc[endpoint], rtol=0, atol=1e-12)
    baseline_summary = anchored.groupby(["endpoint","setting"]).published_factorial_ami.agg(["mean","std"]).sort_index()
    anchored_views = supplied.xs("factorial_k",level="protocol").loc[baseline_summary.index]
    for expected, published in (("mean","published_factorial_ami_mean"),("std","published_factorial_ami_sd")):
        np.testing.assert_allclose(baseline_summary[expected],anchored_views[published],rtol=0,atol=1e-12)
    return supplied, pairs, arch, audit, checks


def overview(summaries, pairs, archive, checks):
    ratios = pairs.pivot(index="endpoint", columns="block", values="between_over_within")
    matches, anchors = checks["match_at_1e_12"], checks["archive_anchors"]
    max_delta = checks["max_abs_ami_delta"]
    notes = [
        "# Pooled-channel weighting and k-means",
        "",
        "The published construction factorial used MiniBatchKMeans with K selected from the full row "
        "count. Request 78 specifies K equal to the reference class count. Both scans use the original "
        "five initialization seeds and MiniBatchKMeans settings on the same 12 full factorial panels. "
        "The D2048 reference and all weight settings were refitted with scikit-learn. "
        "Ten endpoints use sparse float32 matrices. The two TPL endpoints use dense updates "
        "with a packed-bit k-means++ initializer. Its squared distances and selected centres "
        "match native initialization on validation inputs. Both layouts preserve the exact "
        "weighted feature values. The factorial-K rows also provide the archived dense D2048 "
        "and w = 1 AMI for direct comparison. "
        f"Of those {anchors} matched anchors, {matches} agree at 1e-12 AMI. The maximum "
        f"absolute difference is {max_delta:.4f} AMI. Sparse accumulation and the recorded "
        "runtime can affect cluster assignments, so the weighting contrasts compare fits "
        "using a common layout within each endpoint.",
        "",
        "All settings use WL radius 2, parity and fingerprint seed 1. At w = 0 only the 1,024-bit "
        "signed block remains active. D2048 uses all 2,048 bits for signed changes. The distance "
        "decomposition samples 2,000 unordered pairs per type without replacement, conditional on "
        "the reference labels. It uses the same pairs for both blocks. The full-row cohort includes "
        "the original zero placeholders, and the CSV counts sampled pairs involving them.",
        "",
    ]
    lower_u = set(ratios.index[ratios.U < ratios.D])
    loss = set(archive.index[archive < 0])
    notes.extend([
        f"U has the smaller between/within ratio on {len(lower_u)}/12 endpoints and on "
        f"{len(lower_u & loss)}/{len(loss)} endpoints with a negative paper contrast. "
        "Both blocks can carry class structure. Weighting tests whether this distance pattern "
        "translates to higher AMI after clustering.",
        "",
        "In the tables, a loss means the paper’s mean D+U minus D AMI over 18 matched "
        "constructions is negative. 'Zero helps' compares w = 0 with w = 1. 'Monotone' "
        "requires every step toward a smaller weight to preserve or improve mean AMI. "
        "The best weight is descriptive on the same cohort.",
        "",
    ])
    outcomes = {}
    for protocol in PROTOCOLS:
        table = summaries.xs(protocol, level="protocol").ami_mean.unstack("setting")
        zero = table.u0 > table.u1 + 1e-12
        observed=np.asarray(table.loc[:,SETTINGS[1:]])
        monotone = np.all(np.diff(observed, axis=1) <= 1e-12, axis=1)
        monotonics=set(table.index[monotone])
        better=set(table.index[zero])
        best=table.loc[:,SETTINGS[1:]].idxmax(axis=1)
        baseline_loss=set(table.index[table.u1 < table.d2048 - 1e-12])
        outcomes[protocol]={
            "paper_loss_endpoints":len(loss), "zero_helps_on_paper_loss":len(better&loss),
            "monotone_on_paper_loss":len(monotonics&loss),
            "wl_r2_union_loss_endpoints":len(baseline_loss),
            "zero_helps_on_wl_r2_union_loss":len(better&baseline_loss),
            "monotone_on_wl_r2_union_loss":len(monotonics&baseline_loss),
        }
        title="Known K" if protocol=="known_k" else "Factorial K"
        notes.extend([
            f"## {title}",
            "",
            f"On the {len(loss)} paper-loss endpoints, w = 0 exceeds w = 1 on "
            f"{len(better&loss)} and the full scan is monotone on {len(monotonics&loss)}. "
            f"For this WL r2 configuration, w = 1 is below D2048 on {len(baseline_loss)} "
            f"endpoints. Among those, w = 0 improves on w = 1 on {len(better&baseline_loss)}, "
            f"with a monotone scan on {len(monotonics&baseline_loss)}.",
            "",
            "| Endpoint | Paper loss | D2048 AMI | w = 1 AMI | w = 0 AMI | Best w | Zero helps | Monotone | D ratio | U ratio | U ratio lower |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
        ])
        for endpoint,r in table.iterrows():
            q=ratios.loc[endpoint]
            yes=lambda answer: "Yes" if answer else "No"
            notes.append(f"| {endpoint} | {yes(endpoint in loss)} | {r.d2048:.4f} | {r.u1:.4f} | {r.u0:.4f} | "
                         f"{best.loc[endpoint][1:]} | {yes(zero.loc[endpoint])} | {yes(endpoint in monotonics)} | "
                         f"{q.D:.3f} | {q.U:.3f} | {yes(endpoint in lower_u)} |")
        notes.append("")
    notes.extend([
        "See the [weight scan](clustering_weight_scan.csv), [per-seed AMI](clustering_weight_cells.csv), "
        "and [distance ratios](clustering_distance_ratio.csv).",
        "",
    ])
    outcomes["distance_ratios"]={"lower_U_overall":len(lower_u),"lower_U_on_paper_loss":len(lower_u&loss)}
    return "\n".join(notes),outcomes


def main():
    root=LOCAL/"clustering_v4"
    summaries,pairs,arch,audit,checks=check_clustering(root)
    barrier=read(OUTPUT/"barrier_AUDIT.json")
    check_files(OUTPUT,{"barrier_groups.csv":barrier["table_sha256"]})
    for name in ("clustering_weight_scan.csv","clustering_weight_cells.csv",
                 "clustering_distance_ratio.csv","clustering_AUDIT.json"):
        shutil.copyfile(root/name,OUTPUT/name)
    note,counts=overview(summaries,pairs,arch,checks)
    (OUTPUT/"clustering_note.md").write_text(note)
    records=sorted(p for p in OUTPUT.iterdir() if p.name!="COMPLETE.json")
    write(OUTPUT/"COMPLETE.json",{"status":"complete","files_sha256":{p.name:digest(p) for p in records},
                                "code_sha256":{p.name:digest(p) for p in sorted(HERE.iterdir())
                                               if p.suffix in (".py", ".c")},
                                "barrier_totals":{"datasets":9,"rows":barrier["full_parity_feature_replay_rows"]},
                                "clustering_scored_cells":audit["fit_cells"],"clustering_outcomes":counts,
                                "archive_anchor_comparison":checks})
    print(counts)


if __name__=="__main__":
    main()
