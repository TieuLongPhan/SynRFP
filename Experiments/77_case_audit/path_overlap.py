"""Cross-tabulate exact signed classes and attribute the gain from pooled counts."""
from functools import lru_cache

import numpy as np
import pandas as pd

from audit_io import BASE, PATHRELEASE, REQUEST74, export, read


def groups(values):
    return np.unique(values, return_inverse=True)[1]


def majority(group, label):
    n_labels = int(label.max()) + 1
    pairs, counts = np.unique(group * n_labels + label, return_counts=True)
    parent = pairs // n_labels
    top = np.zeros(int(group.max()) + 1, dtype=np.int64)
    np.maximum.at(top, parent, counts)
    return top, np.bincount(parent)


def refinement(parent, child, label, top):
    mapping = np.full(len(np.bincount(child)), -1, dtype=np.int64)
    mapping[child] = parent
    np.testing.assert_array_equal(mapping[child], parent)
    child_top, _ = majority(child, label)
    difference = np.zeros(len(top), dtype=np.int64)
    np.add.at(difference, mapping, child_top)
    difference -= top
    assert (difference >= 0).all()
    return difference


def overlap(path, wl, path_du, label):
    size = np.bincount(path)
    top, n_labels = majority(path, label)
    n_wl = int(wl.max()) + 1
    joint_pair, joint = np.unique(path * n_wl + wl, return_inverse=True)
    n_wl_by_path = np.bincount(joint_pair // n_wl)
    merged = n_wl_by_path > 1
    mixed = n_labels > 1
    single = size == 1
    gain = refinement(path, path_du, label, top)
    wl_gain = refinement(path, joint, label, top)
    assert not gain[~mixed].any() and not gain[single].any()
    rows = int(size[merged].sum())
    return {
        "path_d_classes": len(size),
        "wl_d_classes": n_wl,
        "path_d_singleton_classes": int(single.sum()),
        "path_d_merging_wl_d_classes": int(merged.sum()),
        "path_d_merge_reactions": rows,
        "path_d_merge_reaction_share": rows / len(path),
        "path_d_merge_label_purity_row_weighted": float(top[merged].sum() / rows),
        "path_d_merge_label_purity_class_mean": float((top[merged] / size[merged]).mean()),
        "path_d_merge_mixed_label_classes": int((merged & mixed).sum()),
        "path_d_merge_mixed_label_reactions": int(size[merged & mixed].sum()),
        "path_d_merge_mixed_label_reaction_share_among_merged": float(size[merged & mixed].sum() / rows),
        "path_d_majority_correct": int(top.sum()),
        "path_du_majority_correct": int(top.sum() + gain.sum()),
        "path_du_correct_gain": int(gain.sum()),
        "path_du_correct_gain_from_mixed_label_classes": int(gain[mixed].sum()),
        "path_du_correct_gain_from_singleton_classes": int(gain[single].sum()),
        "path_du_correct_gain_from_path_merges_of_wl_d": int(gain[merged].sum()),
        "path_du_correct_gain_from_other_path_classes": int(gain[~merged].sum()),
        "path_du_correct_gain_share_from_path_merges_of_wl_d": float(gain[merged].sum() / gain.sum()) if gain.sum() else None,
        "path_du_improved_mixed_label_classes": int((gain > 0).sum()),
        "path_plus_wl_d_correct_gain": int(wl_gain.sum()),
    }


def path_reports(audit, output):
    audit.release(PATHRELEASE, REQUEST74, "request_74_results", ["AUDIT.json", "endpoint_summary.csv"])
    released = read(PATHRELEASE / "AUDIT.json")
    table = pd.read_csv(PATHRELEASE / "endpoint_summary.csv")
    old = BASE / "delta_ceiling_request74_v2"
    encoded = read(audit.use(old / "ENCODED.json"))
    config = read(audit.use(old / "ENCODING_PLAN.json", encoded["plan_sha256"]))
    assert config == released["encoding"]
    assert encoded["sources"] == released["cohort_source_receipts"]
    profile_order, channels = config["token_profiles"], config["channels"]
    path_profile, wl_profile = (profile_order.index(name) for name in ("path_r2", "wl_r2"))
    delta, combined = (channels.index(channel) for channel in ("D", "D+U"))
    native = BASE / "default_optimal_benchmark_v1/native/classification/plan.json"
    panels = read(audit.use(native, config["input_plans_sha256"][str(native)]))["panels"]
    assert set(panels) == set(released["endpoints"])
    assert len(panels) == 14

    @lru_cache(maxsize=2)
    def key_archive(source):
        path = audit.use(old / (source + ".npz"), encoded["sources"][source]["artifact_sha256"])
        with np.load(path, allow_pickle=False) as saved:
            keys = saved["keys"]
            return (saved["row_ids"], saved["ok"], keys[:, path_profile, delta].copy(),
                    keys[:, wl_profile, delta].copy(), keys[:, path_profile, combined].copy())

    results = []
    for endpoint, panel in sorted(panels.items()):
        proof = released["endpoints"][endpoint]
        assert panel["table_sha256"] == proof["panel_sha256"]
        selected = pd.read_csv(audit.use(panel["table"], proof["panel_sha256"]),
                               usecols=["row_id", "source_position", "dataset_position", "y"], dtype={"row_id": str})
        np.testing.assert_array_equal(selected.dataset_position, np.arange(panel["n"]))
        ids, ok, *keys = key_archive(panel["source"])
        positions = selected.source_position.to_numpy()
        np.testing.assert_array_equal(selected.row_id, ids[positions])
        mask = ok[positions]
        assert int(mask.sum()) == proof["encoded"]
        label = groups(selected.y.to_numpy()[mask])
        path, wl, path_du = (groups(k[positions[mask]]) for k in keys)
        figures = overlap(path, wl, path_du, label)
        published = table.query("endpoint == @endpoint and tokenizer == 'path_r2'").set_index("channel_set")
        assert figures["path_d_classes"] == published.loc["D", "n_groups"]
        assert len(np.bincount(path_du)) == published.loc["D+U", "n_groups"]
        assert figures["path_d_singleton_classes"] == published.loc["D", "n_singleton_groups"]
        assert len(label) == published.loc["D", "n_reactions"]
        published_wl = table.query("endpoint == @endpoint and tokenizer == 'wl_r2' and channel_set == 'D'").iloc[0]
        assert figures["wl_d_classes"] == published_wl.n_groups
        assert figures["path_d_majority_correct"] == round(published.loc["D", "ceiling"] * len(label))
        assert figures["path_du_majority_correct"] == round(published.loc["D+U", "ceiling"] * len(label))
        delta_mcc = published.loc["D", "published_mlp_context_mcc_change"]
        gain = figures["path_du_correct_gain"]
        results.append({
            "endpoint": endpoint, "n_reactions": len(label), "study65_rows": len(selected),
            "model_context_effect": "gain" if delta_mcc > 0 else "loss" if delta_mcc < 0 else "tie",
            "published_mlp_context_mcc_change": delta_mcc,
            **figures, "path_du_ceiling_gain_pp": 100 * gain / len(label),
        })
    export(output / "path_overlap_summary.csv", results)
    return results
