"""Validate and report the BitBIRCH scan alongside the frozen k-means scan."""
from importlib.metadata import version
import math
import subprocess

import numpy as np
import pandas as pd

from scan import (
    ALLOCATIONS, ANCHORS, HERE, REP78, RELEASE78, REQUEST78, ROOT, SEEDS, THRESHOLD,
    digest, read, write,
)

OUTPUT = HERE / "publication_metadata/clustering_channels_v1"
GROUPS = {
    "reaction_centre": {"mech_1", "mech_2", "syntemp_r0", "uspto50k_b", "uspto50k_ub", "schneider_ub"},
    "context": {"ecreact_1", "ecreact_2", "ecreact_3", "schneider_b", "tpl_b", "tpl_ub"},
}
CLASS = {endpoint: group for group, endpoints in GROUPS.items() for endpoint in endpoints}
SETTINGS = dict(d2048=(2048, 0), u2048=(0, 2048), du1024_1024=(1024, 1024), **ALLOCATIONS)
METRICS = ("ami", "ari", "n_clusters", "singleton_share", "runtime_seconds")


def bound_kmeans(pins):
    root = REP78 / "publication_metadata/clustering_barrier_ceiling_v1"
    path = root / "COMPLETE.json"
    expected = subprocess.check_output(
        ["git", "-C", str(REQUEST78), "show", f"{RELEASE78}:{path.relative_to(REQUEST78)}"]
    )
    assert expected == path.read_bytes()
    seal = read(path)
    assert seal["clustering_scored_cells"] == 720
    for name in ("COMPLETE.json", "clustering_weight_scan.csv", "clustering_weight_cells.csv"):
        p = root / name
        h = digest(p)
        if name != "COMPLETE.json":
            assert h == seal["files_sha256"][name]
        pins[str(p)] = h
    return pd.read_csv(root / "clustering_weight_scan.csv")


def load_cells():
    plan = read(ROOT / "prepared.json")
    inputs = plan["inputs_sha256"].copy()
    inputs[str(ROOT / "prepared.json")] = digest(ROOT / "prepared.json")
    cells = []
    native_n = 0
    for source, receipt in plan["sources"].items():
        folder = ROOT / source
        enc = folder / "encoded.json"
        unequal = read(enc)
        inputs[str(enc)] = digest(enc)
        assert set(unequal["matrices"]) == set(ALLOCATIONS)
        assert unequal["n_encoded"] <= unequal["n_rows"]
        hashes = {**receipt["matrices"], **unequal["matrices"]}
        for k, expected in hashes.items():
            source_file = folder / f"{k}.npy"
            assert digest(source_file) == expected
        for group in sorted({p["group"] for p in receipt["panels"]}):
            panels = {p["endpoint"]: p for p in receipt["panels"] if p["group"] == group}
            for construction in SETTINGS:
                for seed in SEEDS:
                    p = folder / f"{group}__{construction}__{seed}.json"
                    raw = read(p)
                    assert raw["packed_sha256"] == hashes[construction]
                    inputs[str(p)] = digest(p)
                    assert {r["endpoint"] for r in raw["scores"]} == set(panels)
                    for row in raw["scores"]:
                        assert (row["group"], row["construction"], row["order_seed"], row["threshold"]) == (
                            group, construction, seed, THRESHOLD)
                        assert all(row[k] == v for k, v in panels[row["endpoint"]].items())
                        assert 1 <= row["n_clusters"] <= row["n_rows"]
                        assert -1 <= row["ami"] <= 1 and 0 <= row["singleton_share"] <= 1
                        assert row["runtime_seconds"] > 0 and all(math.isfinite(row[x]) for x in METRICS)
                        db, ub = SETTINGS[construction]
                        cells.append(dict(**row, d_bits=db, u_bits=ub))
                    native_n += 1
    f = pd.DataFrame(cells).sort_values(["endpoint", "construction", "order_seed"])
    assert len(f) == 300 and not f.duplicated(["endpoint", "construction", "order_seed"]).any()
    assert set(f.endpoint) == set(CLASS)
    return f, inputs, native_n


def comparison(bb, km):
    assert len(bb) == 60 and len(km) == 144
    records = []
    for method, protocol, frame, full in [
        ("bitbirch", "fixed_threshold", bb, "du1024_1024"),
        ("kmeans", "known_k", km[km.protocol.eq("known_k")], "u1"),
        ("kmeans", "factorial_k", km[km.protocol.eq("factorial_k")], "u1"),
    ]:
        for endpoint, g in frame.groupby("endpoint", sort=True):
            g = g.set_index("setting", verify_integrity=True)
            d_ref, full_ref = g.loc["d2048", "ami_mean"], g.loc[full, "ami_mean"]
            assert (g.n_rows == g.n_rows.iloc[0]).all()
            for setting, r in g.iterrows():
                records.append(dict(method=method, protocol=protocol, group=CLASS[endpoint],
                                    level="endpoint", endpoint=endpoint, setting=setting,
                                    n_endpoints=1, n_rows=int(r.n_rows), ami_mean=r.ami_mean,
                                    ami_delta_vs_d=r.ami_mean-d_ref, ami_delta_vs_full_du=r.ami_mean-full_ref))
    f = pd.DataFrame(records)
    g = f.groupby(["method", "protocol", "group", "setting"], sort=True)
    groups = g[["ami_mean", "ami_delta_vs_d", "ami_delta_vs_full_du"]].mean().reset_index()
    groups["level"] = "group"
    groups["n_endpoints"] = g.size().to_numpy()
    assert groups.n_endpoints.eq(6).all()
    return pd.concat([f, groups], ignore_index=True)


def write_note(cells, summary, native_n):
    grouped = summary[summary.level.eq("group")]
    lines = [
        "# Channel allocation in BitBIRCH and k-means", "",
        f"The scan contains **300 endpoint scores** from **{native_n} native BitBIRCH fits** across all 12 endpoints. "
        "ECReact hierarchy levels reuse the same cluster assignments because their ordered reaction panels are identical. "
        "All published cohort rows and refusal placeholders are retained. "
        "Each construction uses WL radius 2, parity with fingerprint seed 1, and a 2,048-bit budget.", "",
        "BitBIRCH used Tanimoto with the archived SynRFP threshold 0.35, branching factor 50 and diameter merging. "
        "The threshold had been selected by AMI on balanced USPTO50K using the historical 1,024-bit delta fingerprint. "
        "It was fixed across all endpoints and all five constructions in this scan. "
        "The five input-order seeds are 20260924 through 20260928, matched across constructions at each endpoint. "
        "BitBIRCH used no reference class count. The two k-means protocols retain their respective Request 78 K rules.", "",
        "The BitBIRCH table gives arithmetic means and sample standard deviations across input orders. "
        "Singleton share counts reactions in singleton clusters. Runtime times the native fit and assignment extraction. "
        "Group means below give each of the six endpoint views the same weight, including the three related ECReact views.", "",
        "| Method | Pooled setting | Reaction-centre ΔAMI vs D | Context ΔAMI vs D | Reaction-centre recovery vs full D+U | Context recovery vs full D+U |",
        "|---|---|---:|---:|---:|---:|",
    ]
    methods = [("bitbirch", "fixed_threshold", ("du1024_1024", *ALLOCATIONS)),
               ("kmeans", "known_k", ("u1", "u0.75", "u0.5", "u0.25", "u0")),
               ("kmeans", "factorial_k", ("u1", "u0.75", "u0.5", "u0.25", "u0"))]
    verdict = []
    for method, protocol, settings in methods:
        report = grouped[grouped.method.eq(method) & grouped.protocol.eq(protocol)].set_index(["group", "setting"])
        comparisons = []
        for setting in settings:
            rc, ctx = (report.loc[(group, setting)] for group in GROUPS)
            a, b = rc.ami_delta_vs_full_du, ctx.ami_delta_vs_full_du
            name = "BitBIRCH" if method == "bitbirch" else ("k-means, reference K" if protocol == "known_k" else "k-means, factorial K")
            lines.append(f"| {name} | `{setting}` | {rc.ami_delta_vs_d:+.4f} | {ctx.ami_delta_vs_d:+.4f} | {a:+.4f} | {b:+.4f} |")
            if setting != settings[0]:
                comparisons.append((setting, b-a, b))
        better = [setting for setting, gain, ctx_gain in comparisons if gain > 0 and ctx_gain > 0]
        before = settings[0]
        base, ctx_base = (report.loc[(group, before), "ami_delta_vs_d"] for group in GROUPS)
        subset = summary[(summary.level == "endpoint") & (summary.method == method) &
                         (summary.protocol == protocol) & (summary.setting == before)]
        losses = (subset.ami_delta_vs_d < 0).sum()
        detail = f"The context group gained more in {len(better)} of {len(comparisons)} lower-U settings"
        if better:
            detail += " (" + ", ".join(f"`{key}`" for key in better) + ")"
        verdict.append(f"- **{name}**. Full D+U lowered AMI on {losses}/12 endpoints. "
                       f"The group changes versus D were {base:+.4f} for reaction-centre endpoints "
                       f"and {ctx_base:+.4f} for context endpoints. {detail}.")
    lines.extend(["", *verdict, "",
                  "For k-means, `u0` retains a 1,024-bit D block with U weighted to zero. "
                  "The D reference uses 2,048 bits. Unequal BitBIRCH allocations change the fold modulus as well as the share of U bits. "
                  "Their results therefore reflect both effects. AMI changes are descriptive on these fixed panels and at this threshold.", "",
                  "See [BitBIRCH summary](bitbirch_scan.csv), [individual orders](bitbirch_order_scores.csv) and "
                  "[endpoint and group contrasts](group_summary.csv)."])
    (OUTPUT / "clustering_channels_note.md").write_text("\n".join(lines) + "\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells, pins, native_n = load_cells()
    keys = ("endpoint", "construction", "d_bits", "u_bits", "threshold", "n_rows", "n_refusal_placeholders")
    stats = cells.groupby(list(keys), as_index=False)[list(METRICS)].agg(["mean", "std"])
    stats.columns = ["_".join(part for part in c if part) for c in stats.columns]
    stats = stats.rename(columns={"construction": "setting", **{metric + "_std": metric + "_sd" for metric in METRICS}})
    stats["n_orders"] = len(SEEDS)
    km = bound_kmeans(pins)
    n = cells.groupby("endpoint")["n_rows"].first().sort_index()
    for _, frame in km.groupby(["protocol", "setting"]):
        assert frame.set_index("endpoint").n_rows.sort_index().equals(n)
    summary = comparison(stats, km)
    cells.to_csv(OUTPUT / "bitbirch_order_scores.csv", index=False)
    stats.to_csv(OUTPUT / "bitbirch_scan.csv", index=False)
    summary.to_csv(OUTPUT / "group_summary.csv", index=False)
    write_note(cells, summary, native_n)
    write(OUTPUT / "COMPLETE.json", dict(
        status="complete", request_commit="a419562a1221355d16e99ead86d8f7f1417f7372",
        request78_revision=RELEASE78, n_native_fits=native_n, n_endpoint_scores=len(cells),
        n_summary_rows=len(stats), n_contrast_rows=len(summary), order_seeds=SEEDS,
        threshold=THRESHOLD, fingerprints=SETTINGS, bitbirch_branching_factor=50,
        bitbirch_merge_criterion="diameter", fingerprint_seed=1,
        versions={key: version(key) for key in ("bblean", "numpy", "scikit-learn", "rdkit", "synkit", "numba", "pandas")},
        inputs_sha256=pins, code_sha256={name: digest(HERE/name) for name in ("scan.py", "report.py")},
        files_sha256={f.name: digest(f) for f in OUTPUT.iterdir() if f.name != "COMPLETE.json"},
    ))
    print(len(cells), "verified endpoint scores from", native_n, "native fits")


if __name__ == "__main__":
    main()
