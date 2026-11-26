"""Export the 36 original fixed-threshold BitBIRCH results from Git."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REV = "55181ba9058380404144dc70ea5743f53962e3a0"
ENDPOINTS = (
    "schneider_b", "schneider_ub", "tpl_b", "tpl_ub", "uspto50k_b", "uspto50k_ub",
    "syntemp_r0", "mech_1", "mech_2", "ecreact_1", "ecreact_2", "ecreact_3",
)
CONFIGS = {"synrfp-wl-r2-b1024-parity-delta": .35, "drfp": .35, "rdkit_struct": .65}


def main():
    out = HERE / "publication_metadata"
    out.mkdir(exist_ok=True)
    rows = []
    for endpoint in ENDPOINTS:
        level = {"mech": "level", "ecreact": "ec"}.get(endpoint.split("_")[0])
        label = level + endpoint[-1] if level else "label_0" if endpoint == "syntemp_r0" else "label"
        for config, threshold in CONFIGS.items():
            path = f"Experiments/05_clustering/results/{endpoint}/{config}/bitbirch_t{threshold:.2f}.json"
            data = subprocess.check_output(["git", "-C", str(HERE), "show", f"{REV}:{path}"])
            result = json.loads(data)
            assert (result["dataset"], result["method"], result["param"]) == (endpoint, "bitbirch", threshold)
            scores = {metric: result[f"{label}__{metric}"] for metric in
                      ("ami", "ari", "n_clusters", "singleton_fraction")}
            rows.append(dict(endpoint=endpoint, fingerprint=config, threshold=threshold, n_rows=result["n"],
                             **scores, source_commit=REV, source_path=path,
                             source_sha256=hashlib.sha256(data).hexdigest()))
    with (out / "legacy_bitbirch_reference.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
