"""Reproduce the Case F population from saved predictions and report aggregate chemistry."""
from functools import lru_cache
import json
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import chi2_contingency, random_table

from audit_io import BASE, CASERELEASE, REQUEST75, WORKSPACE, export, read

SEEDS = (17, 29, 43)
ARMS = ("d2048", "d_pooled")
QP_TARGETS = ("barrier", "deltaHrxn")
DIRECTIONS = ("gain", "loss")
EC_SUBCLASS = re.compile(r"(?:EC\s*)?([1-9]\d*\.[1-9]\d*)(?:\.[^.]+){0,2}")


def saved_predictions(audit, paths, positions):
    predictions = []
    for path, sha in paths:
        with np.load(audit.use(path, sha), allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["matrix_row"], positions)
            predictions.append(saved["prediction"])
    result = np.stack(predictions)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite archived prediction")
    return result


def enzymes(audit):
    checkout = WORKSPACE / "SynRFP-enzymemap-precheck/Experiments"
    root = checkout / "work/74_context_dependence"
    result = root / "enzymemap-replication-v1"
    status = read(audit.use(result / "status.json"))
    assert status["phase"] == "complete" and status["verified_fits"] == 30
    plan_path = audit.use(result / "plan.json", status["outputs_sha256"]["plan.json"])
    plan = read(plan_path)
    receipts = read(audit.use(result / "fit_receipts.json", status["outputs_sha256"]["fit_receipts.json"]))
    fits = {(r["target"], r["arm"], r["seed"]): r for r in receipts}

    precheck = root / "enzymemap-precheck-v1"
    eligibility = read(audit.use(precheck / "eligibility_precheck.json"))
    cohort = pd.read_csv(audit.use(precheck / "cohort.csv.gz", eligibility["outputs_sha256"]["cohort.csv.gz"]), dtype=str)
    source = checkout / "74_context_dependence/inputs/enzymemap_v2_brenda2023.csv.gz"
    raw = pd.read_csv(audit.use(source, plan["source_sha256"]), usecols=["rxn_idx", "ec_num"], dtype=str)

    def partition(split):
        folder = root / f"enzymemap-labels-{split}-v1"
        receipt = read(audit.use(folder / "COMPLETE.json", plan["map_receipt_sha256"][split]))
        mapped = read(audit.use(folder / f"{split}_labels.json", receipt["labels_sha256"]))
        frame = cohort.loc[cohort.split == split]
        assert frame.row_id.tolist() == [r["row_id"] for r in mapped]
        represented = [(i, r["label"]) for i, r in enumerate(mapped) if "label" in r]
        positions, labels = zip(*represented)
        chosen = frame.iloc[list(positions)].copy()
        assert len(chosen) == plan["partition_admitted"][split]
        records = raw.iloc[chosen.representative_source_index.to_numpy(int)]
        for source_id, cohort_ids in zip(records.rxn_idx, chosen.source_ids, strict=True):
            assert source_id in json.loads(cohort_ids)
        chosen["ec_full"] = [re.sub(r"^EC\s*", "", s.strip()) for s in records.ec_num]
        chosen["ec"] = [EC_SUBCLASS.fullmatch(s)[1] for s in chosen.ec_full]
        chosen["bond"] = labels
        return chosen

    train, test = (partition(split) for split in ("train", "test"))
    start = len(train) + plan["partition_admitted"]["validation"]
    indices = np.arange(start, start + len(test))
    prediction = {}
    for target, field in (("ec_level_2", "ec"), ("bond_edit_multiset", "bond")):
        vocabulary = np.unique(train[field].to_numpy(str))
        for arm in ARMS:
            paths = []
            for seed in SEEDS:
                receipt = fits[target, arm, seed]
                assert receipt["plan_sha256"] == audit.pins[str(plan_path)]
                path = result / "fits" / f"{target}__{arm}__s{seed}" / "blind_predictions.npz"
                paths.append((path, receipt["outputs_sha256"][path.name]))
            prediction[field, arm] = vocabulary[saved_predictions(audit, paths, indices)]
    label = test.ec.to_numpy(str)
    d, du = (prediction["ec", arm] == label for arm in ARMS)
    paired = du.astype(int) - d
    labels = {"ec_main": test.ec.str.split(".").str[0].to_numpy(),
              "ec_subclass": label, "bond_edit": test.bond.to_numpy(str),
              "ec_full": test.ec_full.to_numpy(str)}
    return test.row_id.to_numpy(str), labels, paired, prediction


def pioneer(audit):
    base = BASE / "context_dependence_d421d7ae"
    result = base / "qp_primary_v1"
    status = read(audit.use(result / "status.json"))
    assert status["phase"] == "complete" and status["fits"] == 30
    path = audit.use(result / "manifest.json", status["outputs_sha256"]["manifest.json"])
    manifest = read(path)
    folder = base / "qp_frozen_v1"
    frozen = read(audit.use(folder / "prediction.json", manifest["frozen_prediction_sha256"]))
    with np.load(audit.use(folder / "cohort.npz", frozen["artifact_sha256"]["cohort.npz"]), allow_pickle=False) as saved:
        positions, ids, train = saved["test_indices"], saved["test_ids"], saved["train_targets"]
    source = base / "sources/quantumpioneer_kinetics.csv"
    raw = pd.read_csv(audit.use(source, frozen["source_sha256"]), usecols=[
        "source_record_id", "proposed_split", "reaction_smiles", "energy_unit", *QP_TARGETS,
    ])
    test = raw.iloc[positions]
    np.testing.assert_array_equal(test.source_record_id, ids)
    assert (test.proposed_split == "test").all() and (test.energy_unit == "kcal/mol").all()
    truth = test[list(QP_TARGETS)].to_numpy()
    prediction = {}
    for target in QP_TARGETS:
        for arm in ARMS:
            paths = []
            for seed in SEEDS:
                model = result / "cells" / f"{target}__{arm}__s{seed}"
                receipt = read(audit.use(model / "COMPLETE.json"))
                assert receipt["plan_sha256"] == audit.pins[str(path)]
                pred = model / "blind_predictions.npz"
                paths.append((pred, receipt["outputs_sha256"][pred.name]))
            prediction[target, arm] = saved_predictions(audit, paths, positions)
    primary = frozen["predicted_higher_gain_target"]
    index = QP_TARGETS.index(primary)
    d, du = (prediction[primary, arm] for arm in ARMS)
    paired = ((d - truth[:, index]) ** 2 - (du - truth[:, index]) ** 2) / train[:, index].var()
    return ids, test.reaction_smiles, truth, paired, prediction, primary


def eligibility(paired, require_mean=False):
    selected = {}
    for direction, sign in zip(DIRECTIONS, (1, -1), strict=True):
        mask = (sign * paired > 0).sum(axis=0) >= 2
        if require_mean:
            mask &= sign * paired.mean(axis=0) > 0
        selected[direction] = mask
    assert not (selected["gain"] & selected["loss"]).any()
    return selected


def composition(feature, values, masks):
    labels, index = np.unique(values, return_inverse=True)
    table = np.stack([np.bincount(index[mask], minlength=len(labels)) for mask in masks.values()])
    keep = table.sum(axis=0) != 0
    table, labels = table[:, keep], labels[keep]
    totals = table.sum(axis=1)
    records = [
        {"statistic": feature, "label": label, "gain_n": int(gain), "loss_n": int(loss),
         "gain_share": float(gain / totals[0]), "loss_share": float(loss / totals[1])}
        for label, (gain, loss) in zip(labels, table.T, strict=True)
    ]
    return records, table


def chi_square(table):
    statistic, asymptotic_p, dof, expected = chi2_contingency(table, correction=False)
    # Conditional random tables handle the rare categories without pooling.
    n_draws = 49999
    random = random_table.rvs(table.sum(axis=1), table.sum(axis=0), size=n_draws,
                              random_state=np.random.default_rng(77))
    null = ((random - expected) ** 2 / expected).sum(axis=(1, 2))
    return {
        "pearson_chi_square": float(statistic), "dof": int(dof),
        "asymptotic_p": float(asymptotic_p),
        "conditional_mc_p": (1 + int((null >= statistic - 1e-12).sum())) / (1 + n_draws),
        "conditional_draws": n_draws,
        "cramers_v": float(np.sqrt(statistic / table.sum())),
        "expected_cells_below_five": int((expected < 5).sum()),
    }


@lru_cache(maxsize=20000)
def radical_identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Unparsed saved QuantumPioneer component")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    identity = Chem.MolToSmiles(mol)
    Chem.RemoveStereochemistry(mol)
    radicals = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    if radicals:
        atom, = radicals
        assert atom.GetNumRadicalElectrons() == 1
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        atom.SetNoImplicit(True)
        atom.SetNumRadicalElectrons(0)
        Chem.SanitizeMol(mol)
        assert not any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return identity, bool(radicals), Chem.MolToSmiles(mol)


def abstraction(reaction):
    reactant, product = reaction.split(">>")
    (r_closed, r_rad), (p_closed, p_rad) = [
        sorted((radical_identity(text) for text in side.split(".")), key=lambda item: item[1])
        for side in (reactant, product)
    ]
    assert not r_closed[1] and r_rad[1] and not p_closed[1] and p_rad[1]
    assert r_rad[2] == p_closed[2] and p_rad[2] == r_closed[2]
    return r_rad[0]


def distribution(values):
    lo, mid, hi = np.quantile(values, [0.25, 0.5, 0.75])
    return {"q1": float(lo), "median": float(mid), "q3": float(hi),
            "minimum": float(np.min(values)), "maximum": float(np.max(values))}


def external_reports(audit, output):
    audit.release(CASERELEASE, REQUEST75, "request_75_results", ["case_f.csv", "AUDIT.json"])
    published = read(CASERELEASE / "AUDIT.json")
    archived = published["source_counts"]["F_COMPLETE"]
    examples = pd.read_csv(CASERELEASE / "case_f.csv", dtype={"r_id": str, "true_ec_label": str})
    ids, labels, paired, prediction = enzymes(audit)
    masks = eligibility(paired)
    enzyme_records, counts, tests = [], {}, {}
    for feature in ("ec_main", "ec_subclass", "bond_edit"):
        records, table = composition(feature, labels[feature], masks)
        enzyme_records += records
        counts[feature] = {r["label"]: (r["gain_n"], r["loss_n"]) for r in records}
        if feature != "bond_edit":
            tests[feature] = chi_square(table)

    carb = np.array([value == "5.1.3" or value.startswith("5.1.3.") for value in labels["ec_full"]])
    record = {"statistic": "ec_annotation_detail", "label": "5.1.3"}
    for key, mask in masks.items():
        record[f"{key}_n"] = int(carb[mask].sum())
        record[f"{key}_share"] = float(carb[mask].mean())
    enzyme_records.append(record)

    bond = labels["bond_edit"]
    d, du = (prediction["bond", arm] for arm in ARMS)
    bond_vote = (du == bond).astype(int) - (d == bond)
    bond_flags = {
        "true_bond_label_absent_from_other_group": {
            key: ~np.isin(bond, bond[~mask & (masks["gain"] | masks["loss"])])
            for key, mask in masks.items()
        },
        "no_edit": {key: bond == "no_edit" for key in masks},
        "arm_bond_predictions_differ_in_two_or_more_seeds": {key: ((d != du).sum(axis=0) >= 2) for key in masks},
        "bond_correctness_gain_in_two_or_more_seeds": {key: ((bond_vote > 0).sum(axis=0) >= 2) for key in masks},
        "bond_correctness_loss_in_two_or_more_seeds": {key: ((bond_vote < 0).sum(axis=0) >= 2) for key in masks},
    }
    bond_stats = {}
    for metric, flags in bond_flags.items():
        row = {"statistic": "bond_edit_summary", "label": metric}
        for key, mask in masks.items():
            row[f"{key}_n"] = int(flags[key][mask].sum())
            row[f"{key}_share"] = float(flags[key][mask].mean())
        enzyme_records.append(row)
        bond_stats[metric] = row
    for level, test in tests.items():
        enzyme_records.append({"statistic": "composition_test", "label": level, **test})

    def check_membership(dataset, ids, masks):
        assert len(ids) == archived[dataset]["rows"]
        assert len(np.unique(ids)) == len(ids)
        for direction, mask in masks.items():
            assert int(mask.sum()) == archived[dataset]["counts"][f"eligible_{direction}"]
            show = examples.query("dataset == @dataset and direction == @direction").r_id
            assert len(show) == 5 and set(show) <= set(ids[mask])

    check_membership("enzymemap", ids, masks)
    example_ids = examples.query("dataset == 'enzymemap' and direction == 'loss'").r_id
    enzyme = {"rows": len(ids), "counts": counts, "bond_stats": bond_stats, "tests": tests,
              "carbohydrate_annotations": record,
              "selected_carbohydrate_annotations": int((carb & np.isin(ids, example_ids)).sum()),
              "selected_subclasses": {key: examples.query("dataset == 'enzymemap' and direction == @key").true_ec_label.tolist()
                                      for key in masks},
              "totals": {key: int(mask.sum()) for key, mask in masks.items()}}
    export(output / "enzymemap_composition.csv", enzyme_records)

    ids, reactions, truth, paired, predictions, primary = pioneer(audit)
    masks = eligibility(paired, require_mean=True)
    check_membership("quantumpioneer", ids, masks)
    radicals = np.array([abstraction(reaction) for reaction in reactions])
    qp_records, _ = composition("attacking_radical", radicals, masks)
    qp = {"rows": len(ids), "verified_abstractions": len(radicals), "selection_target": primary,
          "totals": {key: int(mask.sum()) for key, mask in masks.items()},
          "radicals": sorted(qp_records, key=lambda r: -(r["gain_n"] + r["loss_n"])),
          "group_metrics": {}, "examples_barrier": {}}
    show = examples.query("dataset == 'quantumpioneer'")
    for direction, mask in masks.items():
        for target_index, target in enumerate(QP_TARGETS):
            values = truth[mask, target_index]
            measures = {"true_" + target: values}
            for arm, short in zip(ARMS, ("D", "D+U"), strict=True):
                pred = predictions[target, arm][:, mask]
                measures[f"abs_error_{target}_{short}_seedmean_prediction"] = abs(pred.mean(axis=0) - values)
                measures[f"mean_abs_error_{target}_{short}_across_seeds"] = abs(pred - values).mean(axis=0)
            for name, values in measures.items():
                cell = {"statistic": "energy_distribution", "label": name, "direction": direction,
                        "n": len(values), "units": "kcal/mol", **distribution(values)}
                qp_records.append(cell)
                qp["group_metrics"][direction, name] = cell
        illustrated = show.query("direction == @direction")
        for arm, short in (("D", "D"), ("DU", "D+U")):
            error = abs(illustrated[f"pred_{arm}_barrier"] - illustrated.true_barrier)
            qp["examples_barrier"][direction, short] = float(error.median())
    export(output / "quantumpioneer_composition.csv", qp_records)

    for path, sha in published["inputs_sha256"].items():
        if path in audit.pins:
            audit.use(path, sha)
    return enzyme, qp
