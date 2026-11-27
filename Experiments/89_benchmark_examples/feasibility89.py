"""Compare fixed expert-test probabilities and select structure examples."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from common89 import BASE, ORIGINAL, canonical_sides, feature_row, load_npz, read


def mean_threshold_record(pins, folder, receipt, threshold):
    path = pins.use(folder/"calibration_decision.json",receipt["artifacts_sha256"]["calibration_decision.json"])
    cal = read(path)
    assert cal["threshold"] == threshold
    return cal


def compare(pins, used):
    root = BASE/"feasibility_frea_v1"
    source = pins.plan(root)
    psyn = root/"models/synrfp_wl_r2_delta__seed1"
    syncat = BASE/"specialist_references_v1/syncat_feasibility"
    cat_plan = read(pins.use(syncat/"plan.json", read(pins.use(syncat/"plan_seal.json"))["plan_sha256"]))
    cat = syncat/"models/seed1"
    assert cat_plan["parent_plan_sha256"] == pins.hashes[str(root/"plan.json")]
    a, b = [read(pins.use(p/"result.json")) for p in (psyn, cat)]
    assert a["seed"] == b["seed"] == 1 and a["phase"] == b["phase"] == "complete"
    assert a["plan_sha256"] == cat_plan["parent_plan_sha256"]
    assert b["plan_sha256"] == pins.hashes[str(syncat/"plan.json")]
    attempt = Path(a["attempt_directory"])
    decisions, metrics, outputs = [], [], []
    for parent, rec, report in ((attempt,a,a["expert_metrics"]), (cat,b,b["evaluation"]["metrics"]["expert"])):
        decisions.append(report["threshold"])
        mean_threshold_record(pins,parent,rec,decisions[-1])
        probs = load_npz(pins.use(parent/"probabilities.npz",rec["artifacts_sha256"]["probabilities.npz"]))
        outputs.append(probs)
        metrics.append(report)
    refs = {}
    for field in ("input_table", "expert_outcomes", "expert_annotations_not_features"):
        p = source[field]
        sha = source["population_files"][p]
        assert cat_plan["inputs_sha256"][p] == sha
        refs[field] = pd.read_csv(pins.use(p, sha),keep_default_na=False)
    rows_path = source["rows"]
    assert source["population_files"][rows_path] == cat_plan["inputs_sha256"][rows_path]
    full = load_npz(pins.use(rows_path, source["population_files"][rows_path]))
    saved = load_npz(pins.use(attempt/"selection.npz", a["artifacts_sha256"]["selection.npz"]))
    for k, r in full.items():
        np.testing.assert_array_equal(saved[k],r)
    selection = full["expert"]
    selected = refs["input_table"].iloc[selection].reset_index(drop=True)
    truth = refs["expert_outcomes"]
    notes = refs["expert_annotations_not_features"]
    for frame in (truth, notes):
        np.testing.assert_array_equal(selected.row_id, frame.row_id)
    shard_rows = [r for s in cat_plan["shards"] if s["partition"] == "expert" for r in s["row_ids"]]
    assert shard_rows == list(selected.row_id)
    assert len(truth) == 751 and truth.y.sum() == 341
    y = truth.y.to_numpy()
    calc = load_npz(pins.use(source["fit_targets"], source["population_files"][source["fit_targets"]]))
    from common89 import module
    scorer = module("request89_frea_metrics",pins.use(ORIGINAL/"Experiments/24_feasibility/metrics.py"))
    preds, prob = [], []
    cal_labels = calc["y_calibration"]
    for result, output, cutoff in zip(metrics, outputs, decisions, strict=True):
        assert scorer.calibration_threshold(cal_labels,output["calibration"]) == cutoff
        actual = scorer.scores(y,output["expert"],cutoff)
        for key in ("auroc","balanced_accuracy","mcc","tn","fp","fn","tp"):
            np.testing.assert_allclose(actual[key],result[key],rtol=0,atol=1e-12)
        prob.append(output["expert"])
        preds.append(output["expert"] >= cutoff)
    s_ok, c_ok = [pred == y for pred in preds]
    cells = dict(both_right=s_ok & c_ok,only_synrfp_right=s_ok & ~c_ok,
                 only_syncat_right=~s_ok & c_ok,both_wrong=~s_ok & ~c_ok)
    outcome = [dict(endpoint="frea_expert",seed=1,expert_label=label,
                    outcome=kind,n_rows=int(((y == label) & mask).sum()),n_label_rows=int((y == label).sum()))
               for label in (0,1) for kind,mask in cells.items()]
    assert sum(r["n_rows"] for r in outcome) == 751
    raw = root/"raw/FREA-benchmark/test.csv"
    expected = source["raw_files"]["expert"]["sha256"]
    source_rows = pd.read_csv(pins.use(raw,expected),keep_default_na=False)
    assert len(source_rows)==len(y)
    examples = []
    for kind, limit in (("only_synrfp_right",4),("only_syncat_right",2)):
        pool = []
        for index in np.flatnonzero((y == 0) & cells[kind]):
            r = selected.iloc[index]
            expert = notes.iloc[index]
            row = feature_row("frea_expert", "", r.row_id, r.rxn, int(selection[index]), label=0)
            author = source_rows.iloc[int(r.row_id.split(":")[-1])]
            released = f"{author.reactant}>>{author['product']}"
            assert author.failure_reasons == expert.failure_reasons
            if re.search(r":\d+\]", released) and canonical_sides(released) == canonical_sides(r.rxn):
                row.update(mapped_rxn=released,mapping_origin="author FREA source")
            row.update(expert_failure_reason=expert.failure_reasons,
                       expert_additional_note=expert.additional_note,expert_difficulty=expert.difficulty,
                       outcome=kind, seed=1,synrfp_probability=float(prob[0][index]),
                       syncat_probability=float(prob[1][index]),synrfp_threshold=decisions[0],
                       syncat_threshold=decisions[1],synrfp_prediction=int(preds[0][index]),
                       syncat_prediction=int(preds[1][index]))
            pool.append(row)
        accepted = 0
        for row in sorted(pool,key=lambda x:(not bool(x["expert_failure_reason"]),x["n_heavy_atoms"],x["r_id"],x["source_position"])):
            if row["scaffold"] in used:
                continue
            accepted += 1
            row["pair_id"] = f"A_{kind}_{accepted:02d}"
            used.add(row["scaffold"])
            examples.append(row)
            if accepted == limit:
                break
    return examples,outcome
