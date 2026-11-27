"""Count empirical label errors at each WL-radius-2 representation stage."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from common86 import BASE, D, DU, HERE, P22, bound, read, write

ENDPOINTS = ("ecreact_1", "mech_2")
COUNTS = BASE / "delta_ceiling_request74_v2"
STUDY65 = BASE / "default_optimal_benchmark_v1"
NATIVE = HERE.parents[2] / "SynRFP-delta-ceiling/Experiments/74_delta_ceiling"
REQUEST74 = NATIVE / "publication_metadata/delta_ceiling_v1"
RESULT = HERE / "publication_metadata/error_floor_ec_mech_v1"
PINS = {}
ENCODER = {}


def pinned(path, expected=None):
    PINS[str(path)] = bound(path, expected)
    return path


def groups(keys):
    return np.unique(keys, return_inverse=True)[1]


def byte_keys(matrix):
    return np.ascontiguousarray(matrix).view(f"V{matrix.shape[1]}").ravel()


def stats(classes, labels):
    n = len(labels)
    n_classes = int(classes.max()) + 1
    n_labels = int(labels.max()) + 1
    pairs, frequencies = np.unique(classes * n_labels + labels, return_counts=True)
    top = np.zeros(n_classes, dtype=int)
    np.maximum.at(top, pairs // n_labels, frequencies)
    majority = int(top.sum())
    mixed = np.bincount(pairs // n_labels, minlength=n_classes) > 1
    sizes = np.bincount(classes, minlength=n_classes)
    return dict(encoding_classes=n_classes, mixed_label_classes=int(mixed.sum()),
                mixed_label_rows=int(sizes[mixed].sum()), majority_correct=majority,
                forced_errors=n-majority, error_floor_pct=100*(n-majority)/n)


def merge_stats(parent, child, parent_labels, child_labels):
    """Verify the sketch is constant within each incoming exact-count class."""
    lookup = np.empty(len(np.unique(parent)), dtype=int)
    lookup[parent] = child
    np.testing.assert_array_equal(lookup[parent], child)
    return dict(classes_merged=parent_labels["encoding_classes"]-child_labels["encoding_classes"],
                added_forced_errors=child_labels["forced_errors"]-parent_labels["forced_errors"])


def support_batch(texts):
    native, count = ENCODER["native"], ENCODER["count"]
    tok = ENCODER["tokenizer"]
    output = np.empty((len(texts), 7), dtype="V32")
    for i, text in enumerate(texts):
        reaction = native._reaction(text, "classification")
        r = tok.tokens_graph(reaction.reactant, 2)
        p = tok.tokens_graph(reaction.product, 2)
        d, u, _ = native.token_algebra(r, p)
        # The unweighted native encoder emits each nonzero token once.
        # Sign-tagged D support has the same equality as the {-1,+1} coefficients.
        sign = count.signed_digest({token: 1 if value > 0 else -1 for token, value in d.items()})
        present = count.signed_digest(dict.fromkeys(u, 1))
        combined = hashlib.sha256(sign+present).digest()
        output[i] = [*count.keys(r, p), sign, present, combined]
    return output


def supports(texts, keys, combined, jobs):
    labels, pick, inverse = np.unique(combined, return_index=True, return_inverse=True)
    assert np.array_equal(labels[inverse], combined)
    representatives = texts[pick]
    batches = [representatives[i:i+256] for i in range(0, len(pick), 256)]
    print(f"Replaying {len(pick):,} exact D+U representatives", flush=True)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as pool:
        values = np.concatenate(list(pool.map(support_batch, batches)))
    # Exact D+U jointly determines both side counts. Validate the inherited exact
    # D and U keys on every row before extending the support to its class members.
    np.testing.assert_array_equal(values[inverse, :4], keys)
    return values[inverse, 4:]


def endpoint(name, study22, study65, count_receipt, profile_i, index, published, jobs):
    panel = study65["panels"][name]
    source = panel["source"]
    spec = study22["sources"][source]
    table = pd.read_csv(pinned(spec["table"], study22["pinned_files"][spec["table"]]), dtype={"row_id": str})
    frame = pd.read_csv(pinned(panel["table"], panel["table_sha256"]), dtype={"row_id": str})
    assert len(frame) == panel["n"]
    np.testing.assert_array_equal(frame.dataset_position, np.arange(len(frame)))
    np.testing.assert_array_equal(table.source_position, np.arange(len(table)))
    pos = frame.source_position.to_numpy()
    np.testing.assert_array_equal(frame.row_id, table.row_id.to_numpy()[pos])
    receipt = count_receipt["sources"][source]
    assert receipt["source_sha256"] == PINS[spec["table"]]
    with np.load(pinned(COUNTS/f"{source}.npz", receipt["artifact_sha256"]), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["row_ids"], table.row_id.to_numpy(str))
        valid = archive["ok"][pos]
        assert receipt["encoded"] == int(archive["ok"].sum())
        selected = pos[valid]
        keys = archive["keys"][selected, profile_i]
    n = int(valid.sum())
    y = groups(frame.y.to_numpy()[valid])
    source_rxn = table.rxn.to_numpy()[selected]
    groupings = {"source_reaction_text": groups(source_rxn)}
    for channel in ("D", "U", "D+U", "D+S"):
        groupings[channel+"_exact"] = groups(keys[:, index[channel]])
    support = supports(source_rxn, keys, keys[:, index["D+U"]], jobs)
    for j, channel in enumerate(("D", "U", "D+U")):
        groupings[channel+"_support"] = groups(support[:, j])
    features = read(pinned(spec["feature_receipt"]))
    for channel, method in (("D", D), ("U", D.replace("parity-delta", "parity-union")), ("D+U", DU)):
        path = pinned(spec["features"][method], features["files"][f"{method}.npz"])
        if channel == "D":
            assert PINS[str(path)] == receipt["feature_replay_sha256"]["wl_r2"]
        if channel == "D+U":
            assert PINS[str(path)] == study65["sources"][source]["feature_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["source_positions"], table.source_position)
            np.testing.assert_array_equal(archive["row_ids"], table.row_id.to_numpy(str))
            np.testing.assert_array_equal(archive["ok"][pos], valid)
            assert int(archive["n_bits"]) == 2048
            matrix = archive["X"][selected]
            assert matrix.dtype == np.uint8 and matrix.shape == (n, 256)
        groupings[channel+"_parity2048"] = groups(byte_keys(matrix))
        if channel == "D+U":
            groupings["D_parity1024"] = groups(byte_keys(matrix[:, :128]))
            groupings["U_parity1024"] = groups(byte_keys(matrix[:, 128:]))
    values = {name: stats(partition, y) for name, partition in groupings.items()}
    assert merge_stats(groupings["D+U_exact"], groupings["D+S_exact"],
                       values["D+U_exact"], values["D+S_exact"])["classes_merged"] == 0
    for channel in ("D", "U", "D+U", "D+S"):
        exact = values[channel+"_exact"]
        previous = published.loc[(name, "wl_r2", channel)]
        assert (n, exact["encoding_classes"]) == (previous.n_reactions, previous.n_groups)
        np.testing.assert_allclose(exact["majority_correct"]/n, previous.ceiling, rtol=0, atol=1e-14)
        np.testing.assert_allclose(exact["mixed_label_rows"]/n, previous.collision_share, rtol=0, atol=1e-14)
        merge_stats(groupings["source_reaction_text"], groupings[channel+"_exact"],
                    values["source_reaction_text"], exact)
    for stage, value in values.items():
        channel, kind = stage.rsplit("_", 1)
        if kind != "support" and not kind.startswith("parity"):
            continue
        exact = channel+"_exact"
        parent = exact if kind == "support" else channel+"_support"
        value.update(merge_stats(groupings[parent], groupings[stage], values[parent], value))
        value["added_forced_errors_vs_exact"] = value["forced_errors"]-values[exact]["forced_errors"]
        assert value["classes_merged"] >= 0 and value["added_forced_errors"] >= 0
    context = dict(endpoint=name, panel_rows=len(frame), encoded_rows=n, encoding_refusals=int((~valid).sum()))
    long = [dict(**context, stage=k, **v) for k, v in values.items()]
    wide = dict(**context)
    for stage, value in values.items():
        if stage == "D+S_exact":
            continue
        wide[stage+"_forced_errors"] = value["forced_errors"]
        wide[stage+"_error_floor_pct"] = value["error_floor_pct"]
        if "added_forced_errors" in value:
            wide[stage+"_added_forced_errors"] = value["added_forced_errors"]
            wide[stage+"_added_forced_errors_vs_exact"] = value["added_forced_errors_vs_exact"]
            wide[stage+"_classes_merged"] = value["classes_merged"]
    return wide, long


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    RESULT.mkdir(parents=True, exist_ok=True)
    study22 = read(pinned(P22/"plan.json", read(P22/"plan_seal.json")["sha256"]))
    study65 = read(pinned(STUDY65/"plan.json", read(STUDY65/"plan_seal.json")["sha256"]))
    plan = STUDY65/"native/classification/plan.json"
    study65 = read(pinned(plan, study65["inputs_sha256"][str(plan)]))
    request74 = read(pinned(COUNTS/"ENCODED.json"))
    count_plan = read(pinned(COUNTS/"ENCODING_PLAN.json", request74["plan_sha256"]))
    for path in (P22/"plan.json", plan):
        assert PINS[str(path)] == count_plan["input_plans_sha256"][str(path)]
    for path, expected in count_plan["files_sha256"].items():
        pinned(path, expected)
    sys.path.insert(0, str(NATIVE))
    import encode_counts as count
    native = count.F
    ENCODER.update(native=native, count=count, tokenizer=native._tokenizer("wl", native.NODE_ATTRS, native.EDGE_ATTRS))
    pin_summary = read(pinned(REQUEST74/"COMPLETE.json"))
    summary = REQUEST74/"endpoint_summary.csv"
    published = pd.read_csv(pinned(summary, pin_summary["files_sha256"][summary.name]))
    published = published.set_index(["endpoint", "tokenizer", "channel_set"])
    assert published.index.is_unique
    profile_i = count_plan["token_profiles"].index("wl_r2")
    index = {channel: i for i, channel in enumerate(count_plan["channels"])}
    wide, long = [], []
    for name in ENDPOINTS:
        row, stages = endpoint(name, study22, study65, request74, profile_i, index, published, args.jobs)
        wide.append(row)
        long.extend(stages)
    pd.DataFrame(wide).to_csv(RESULT/"table_s6_ec_mech.csv", index=False)
    pd.DataFrame(long).to_csv(RESULT/"error_floor_by_stage.csv", index=False)
    lines = [
        "# Classification error floors on ECReact L1 and Mech L2",
        "",
        "The denominator is the shared valid subset of each full study 65 endpoint. "
        "The encoder mask and labels are identical at every stage. The first stage groups "
        "by the exact archived reaction text. Exact D, U and D+U group by the full WL radius 2 "
        "token counts before sketching and replay Request 74's published results. The support "
        "stage retains D sign and U presence while dropping count magnitudes. "
        "The last stage groups by exact bytes of the released parity features with seed 1. "
        "The combined feature uses 1,024 bits per channel.",
        "",
        "Within each class, the best constant prediction is the majority label. The error floor is "
        "the sum of the minority counts divided by the number of encoded rows. These are empirical "
        "error floors on this cohort. A sketch can merge classes of the same label "
        "without increasing this floor.",
        "",
        "| Endpoint | Encoded / total | Source text | Exact D | D support | D folded | Exact D+U | D+U support | D+U folded |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in wide:
        values = [r[f"{x}_error_floor_pct"] for x in ("source_reaction_text", "D_exact", "D_support", "D_parity2048",
                                                     "D+U_exact", "D+U_support", "D+U_parity2048")]
        lines.append(f"| {r['endpoint']} | {r['encoded_rows']:,} / {r['panel_rows']:,} | " +
                     " | ".join(f"{v:.6f}%" for v in values)+" |")
    lines.extend(["", "The detailed CSV gives incremental errors at each transition, the number of "
                  "merged classes, and measurements for pooled-only and 1,024-bit channels."])
    (RESULT/"README.md").write_text("\n".join(lines)+"\n")
    outputs = {p.name: bound(p) for p in RESULT.iterdir() if p.name != "COMPLETE.json"}
    write(RESULT/"COMPLETE.json", dict(status="complete", endpoints=list(ENDPOINTS),
          native_request74_commit="c2f046a9724a70d4821e16d5a6b27b546003f8fe",
          request86_commit="3fdd44463fe4cdec76a92e62ecc6245474c650f6",
          inputs_sha256=PINS, outputs_sha256=outputs,
          code_sha256={p.name: bound(p) for p in (HERE/"common86.py", HERE/"error_floor86.py")},
          packages={package: version(package) for package in ("numpy", "pandas")}))
    print(RESULT/"table_s6_ec_mech.csv")


if __name__ == "__main__":
    main()
