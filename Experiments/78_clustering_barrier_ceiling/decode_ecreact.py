"""Decode Request 75's saved ECReact labels from the curator's pinned upstream table."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.metadata
from pathlib import Path
import platform
import re
import subprocess

import numpy as np
import pandas as pd

from common78 import BASE, HERE, ORIGINAL, OUTPUT, WORKSPACE, Pins, digest, export, read, write

REQUEST = "509e0ff98574fe6573429dc7895effefe0efb818"
REQUEST74 = "c2f046a9"
REQUEST75 = "9e2c2de9"
SYNCAT = "051e7bf9eca68067b6f8d47f7f29f3e855cc7c74"
SYN_RXN = "e7c03766bd6a405da468e281d922cf1beccdd56a"
SRC_URL = f"https://raw.githubusercontent.com/phuocchung123/SynCat/{SYNCAT}/Data/raw/claire_full.csv.gz"
CURATION_URL = f"https://github.com/TieuLongPhan/SynRXN/blob/{SYN_RXN}/script/build_classification_dataset.py#L601"
CURATION_SHA = "0bafed7aa02c09d3cb9ad200251ec3ced668888d5271b35048a75366d53792a1"
SRC_SHA = "cc3731eb2cacd598e2f096fd329460295408f65d0e818cd0a747c9fc6d9088c8"
RELEASE_SHA = "c730e5bd196dc69b5ebc421cce37010162f7e129f149b28196f0069ee242bdcd"
RELEASE_FILE = "Experiments/work/synrxn_cache/artifacts/zenodo/22045145/classification/ecreact.csv.gz"
ARCHIVE74 = WORKSPACE / "SynRFP-delta-ceiling"
ARCHIVE75 = WORKSPACE / "SynRFP-chemistry-cases"
FOLDER74 = "Experiments/74_delta_ceiling/publication_metadata/delta_ceiling_v1"
FOLDER75 = "Experiments/75_chemistry_cases/publication_metadata/chemistry_cases_v1"


def git_bytes(repo, revision, relative):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{relative}"])


def bind_release(pins, repo, revision, folder, files):
    seal = repo / folder / "COMPLETE.json"
    commit_pin = sha256(git_bytes(repo, revision, f"{folder}/COMPLETE.json")).hexdigest()
    manifest = read(pins.use(seal, commit_pin))
    for name in files:
        pins.use(repo / folder / name, manifest["files_sha256"][name])
    return manifest


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def source_maps(pins, upstream, release, curation):
    pins.use(curation, CURATION_SHA)
    source = read_csv(pins.use(upstream, SRC_SHA))
    table = read_csv(pins.use(release, RELEASE_SHA))
    assert len(source) == len(table) == 185734
    # Validate the full SynRXN projection in order. IDs repeat even though
    # source positions and splits still identify the original annotated rows.
    replayed = {
        "r_id": "ecreact_" + (source["index"].astype(int) + 1).astype(str),
        "rxn": source["rxn_smiles"],
        "split": source["split"],
        "orig_index": source["Unnamed: 0"],
    }
    vocabularies, exceptions = {}, {}
    for level in (1, 2, 3):
        ec = f"ec{level}"
        encoded = f"{ec}_encode"
        replayed[ec] = source[encoded]
        distinct = source[[encoded, ec]].drop_duplicates()
        assert distinct[encoded].str.fullmatch(r"\d+").all()
        assert distinct[encoded].is_unique and distinct[ec].is_unique
        vocabulary = dict(zip(distinct[encoded], distinct[ec], strict=True))
        assert len(vocabulary) == {1: 8, 2: 63, 3: 176}[level]
        pattern = r"\d+" + r"(?:\.\d+)" * (level - 1)
        exceptions[ec] = {index: value for index, value in vocabulary.items()
                          if not re.fullmatch(pattern, value)}
        vocabularies[level] = vocabulary
    for name, column in replayed.items():
        np.testing.assert_array_equal(column, table[name], err_msg=f"Projection changed {name}")
    np.testing.assert_array_equal(source.ec1, source.ec2.str.split(".").str[0])
    np.testing.assert_array_equal(source.ec2, source.ec3.str.rsplit(".", n=1).str[0])
    return table, vocabularies, exceptions


def check_row(source, labels, position, row_id, reaction, label, level):
    pos = int(position)
    if pos < 0 or pos >= len(source):
        raise ValueError(f"Out-of-range source position {pos}")
    key = f"ec{level}"
    assert (source.at[pos, "row_id"], source.at[pos, "rxn"]) == (row_id, reaction)
    assert labels.at[pos, key] == label


def cases_a(source, labels, vocabulary):
    cases = read_csv(ARCHIVE75 / FOLDER75 / "case_a.csv")
    selected = cases.loc[cases.endpoint.str.startswith("ecreact_")].to_dict("records")
    fields = {"true_label": "true_ec", "pred_D": "pred_D_ec", "pred_DU": "pred_DU_ec",
              "collision_label": "collision_ec"}
    for row in selected:
        level = int(row["endpoint"][-1])
        check_row(source, labels, row["source_position"], row["r_id"], row["rxn"], row["true_label"], level)
        check_row(source, labels, row["collision_source_position"], row["collision_r_id"],
                  row["collision_rxn"], row["collision_label"], level)
        row["ec_level"] = level
        row.update({decoded: vocabulary[level][row[encoded]] for encoded, decoded in fields.items()})
    assert Counter(row["ec_level"] for row in selected) == {1: 12, 2: 12, 3: 12}
    return selected


def cases_c(source, labels, vocabulary):
    cases = read_csv(ARCHIVE75 / FOLDER75 / "case_c.csv")
    selected = cases.loc[cases.dataset.eq("ecreact")].copy()
    assert len(selected) == 80
    assert not {"pred_D", "pred_DU"}.intersection(selected)
    assert not selected.duplicated(["pair_id", "side"]).any()
    lookup = selected.set_index(["pair_id", "side"])
    output = selected.to_dict("records")
    vocab = vocabulary[3]
    for row in output:
        check_row(source, labels, row["source_position"], row["r_id"], row["rxn"], row["label"], 3)
        assert row["side"] in {"1", "2"}
        peer = lookup.loc[(row["pair_id"], "2" if row["side"] == "1" else "1")]
        assert row["direction"] == peer.direction
        row.update(ec_level=3, true_ec=vocab[row["label"]], pred_D_ec="", pred_DU_ec="",
                   collision_label=peer.label, collision_ec=vocab[peer.label])
    return output


def revision(ref):
    return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", ref], text=True).strip()


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--upstream", type=Path, required=True)
    args.add_argument("--curation", type=Path, required=True)
    args.add_argument("--release", type=Path, default=ORIGINAL / RELEASE_FILE)
    args.add_argument("--output", type=Path, default=OUTPUT / "ecreact_labels")
    options = args.parse_args()
    pins = Pins()
    bind_release(pins, ARCHIVE74, REQUEST74, FOLDER74, ["AUDIT.json"])
    bind_release(pins, ARCHIVE75, REQUEST75, FOLDER75, ["case_a.csv", "case_c.csv"])
    proof = read(ARCHIVE74 / FOLDER74 / "AUDIT.json")
    assert proof["ec_raw_sha256"] == RELEASE_SHA
    labels, vocabulary, exceptions = source_maps(pins, options.upstream, options.release, options.curation)
    source = read_csv(pins.use(BASE / "all_endpoint_factorial_full_v1/sources/ecreact.csv.gz",
                              proof["cohort_source_receipts"]["ecreact"]["source_sha256"]))
    np.testing.assert_array_equal(source.source_position.astype(int), np.arange(len(labels)))
    np.testing.assert_array_equal(source.row_id, labels.r_id)
    a = cases_a(source, labels, vocabulary)
    c = cases_c(source, labels, vocabulary)
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    export(output / "ecreact_label_map.csv", [
        {"level": level, "index": code, "ec": value}
        for level, mapping in vocabulary.items() for code, value in sorted(mapping.items(), key=lambda item: int(item[0]))
    ])
    export(output / "case_a_ec.csv", a)
    export(output / "case_c_ec.csv", c)
    write(output / "LABEL_AUDIT.json", {
        "request_commit": REQUEST,
        "prior_experiment_78_commit": revision("4ba01b4b"),
        "request74_release_commit": revision(REQUEST74),
        "request75_release_commit": revision(REQUEST75),
        "syncat_source_commit": SYNCAT,
        "syncat_source_url": SRC_URL,
        "synrxn_curation_commit": SYN_RXN,
        "synrxn_curation_url": CURATION_URL,
        "synrxn_release": "https://zenodo.org/records/22045145",
        "input_sha256": pins.hashes,
        "runner_sha256": digest(__file__),
        "helper_sha256": digest(HERE / "common78.py"),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas")},
        "source_projection_rows_verified": len(labels),
        "mapping_counts": {f"ec{level}": len(v) for level, v in vocabulary.items()},
        "source_literals_preserved": exceptions,
        "case_a_ec_rows": len(a),
        "case_c_ec_rows": len(c),
        "case_c_prediction_status": "No predictions in archived Case C. Fields empty.",
        "new_fits": 0,
    })
    print("Verified", len(labels), "rows, map sizes", [len(v) for v in vocabulary.values()],
          "and decoded", len(a), "Case A rows and", len(c), "Case C rows.")


if __name__ == "__main__":
    main()
