"""Summarize completed TPL XGBoost cells without treating them as final."""
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
LOGROOT = ROOT.parent / "03_classification" / "results" / "cpu_parallel_finish"
SYN = "synrfp-wl-r2-b1024-parity-delta"
BASELINES = ["drfp", "rdkit_diff", "rdkit_struct", "rxnfp", "rxnfp_mlm"]


def read(ds):
    path = LOGROOT / f"xgb_gpu_{ds}.log"
    rows = []
    pat = re.compile(r"INFO (\w+) cv s(\d+) (.+?)\s+xgb\s+MCC=([-0-9.]+)")
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            rows.append((int(m.group(2)), m.group(3).strip(), float(m.group(4))))
    return pd.DataFrame(rows, columns=["split", "fingerprint", "mcc"])


def main():
    summaries = []
    lines = [
        "# Partial TPL XGBoost analysis",
        "",
        "This is a live, descriptive summary of completed log cells. It must not",
        "be used as the final TPL ranking until both workers finish all 25 splits",
        "and the completeness audit passes.",
        "",
        "| Dataset | Complete splits | Baseline | SynRFP wins | Losses | Mean MCC margin |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for ds in ["tpl_ub", "tpl_b"]:
        d = read(ds).pivot(index="split", columns="fingerprint", values="mcc")
        for base in BASELINES:
            if SYN not in d or base not in d:
                continue
            x = (d[SYN] - d[base]).dropna()
            summaries.append({"dataset": ds, "baseline": base, "margin": x.mean()})
            lines.append(
                f"| {ds} | {len(x)} | {base} | {(x > 0).sum()} | "
                f"{(x < 0).sum()} | {x.mean():+.4f} |"
            )
    lines += [
        "",
        "The completed cells already show RDKit difference ahead of SynRFP on",
        "both TPL views, with DRFP also ahead on the available matched splits.",
        "This is a useful stress-case explanation, but the final table must be",
        "regenerated after the remaining TPL cells complete.",
    ]
    (ROOT / "TPL_PARTIAL_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    s = pd.DataFrame(summaries)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    baselines = BASELINES
    x = range(len(baselines))
    for offset, ds in [(-0.19, "tpl_ub"), (0.19, "tpl_b")]:
        g = s[s.dataset.eq(ds)].set_index("baseline").reindex(baselines)
        ax.bar([i + offset for i in x], g.margin, width=0.35, label=ds)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x), ["DRFP", "RDKit diff", "RDKit struct", "RXNFP", "RXNFP-MLM"])
    ax.set_ylabel("SynRFP − baseline MCC")
    ax.set_title("Partial TPL XGBoost margins (provisional)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(RES / "tpl_partial_margins.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
