"""Replay the released full-row yield metrics from saved predictions."""
import csv
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1] / "Experiments/65_default_optimal/publication_metadata/default_optimal_v1"
ROOT = Path.home() / ".local/share/synrfp-runs/yield_fullrow_v1"
AUTHOR = Path("/var/tmp/synrfp-labhhc4-storage/bh_hte_2026/data/BH_HTE_Curated_Dataset_v20260219.csv")
AUTHOR_SHA = "247473787c33fb22ee48a4177593188f2a2c6ef140756566fe4610fee89ba333"
METHODS = ("synrfp_default", "synrfp_optimal", "drfp", "rdkit_struct")
PAIR = ["Aryl SMILES", "Amine SMILES"]
RECIPE = ["Catalyst SMILES", "Base SMILES", "Solvent SMILES"]
PARTS = ("train", "validation", "test")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Frozen input contents changed: {path}")
    return path


def audit_predictions(ref, selectors, cohort):
    path = verify(Path(ref.source_result_path), ref.source_result_sha256)
    result = read(path)
    assert result["phase"] == "complete"
    if "evaluation_path" in result:
        rows, predictions = (Path(result[key]) for key in ("rows_path", "evaluation_path"))
    else:
        attempt = Path(result["attempt_directory"])
        rows, predictions = attempt / "rows.npz", attempt / "predictions.npy"
    hashes = result["artifacts_sha256"]
    for p in (rows, predictions):
        verify(p, hashes[str(p)])
    test = selectors["test"]
    observed = cohort.yield_pct.to_numpy()[test]
    with np.load(rows, allow_pickle=False) as saved:
        for label in PARTS:
            np.testing.assert_array_equal(saved[label], selectors[label])
    if predictions.suffix == ".npz":
        with np.load(predictions, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["source_rows"], test)
            np.testing.assert_array_equal(saved["row_ids"], cohort.row_id.to_numpy()[test])
            np.testing.assert_array_equal(saved["observed"], observed)
            predicted = saved["prediction"]
    else:
        predicted = np.load(predictions, allow_pickle=False)
    assert predicted.shape == observed.shape and np.isfinite(predicted).all()
    error = predicted - observed
    metrics = dict(r2=1-np.sum(error**2)/np.sum((observed-observed.mean())**2),
                   rmse=np.sqrt(np.mean(error**2)), mae=np.mean(np.abs(error)))
    counts = dict(ntrain=len(selectors["train"]), nval=len(selectors["validation"]), ntest=len(test))
    assert tuple(counts.values()) == (ref.ntrain, ref.nval, ref.ntest)
    record = dict(dataset=ref.dataset, split=ref.split, method=ref.method,
                  configuration=ref.configuration, **counts, result_sha256=ref.source_result_sha256,
                  selector_sha256=hashes[str(rows)], predictions_sha256=hashes[str(predictions)])
    for name, value in metrics.items():
        released = getattr(ref, name)
        np.testing.assert_allclose(value, released, atol=1e-10, rtol=0)
        record.update({name:value, f"{name}_released":released, f"{name}_difference":value-released})
    return record


def pair_coverage(raw, selectors):
    test = raw.iloc[selectors["test"]].drop_duplicates(PAIR+RECIPE)
    counts = test.groupby(PAIR, dropna=False).size()
    eligible = counts.index[counts >= 8]
    non_test = np.concatenate([selectors["train"], selectors["validation"]])
    seen = pd.MultiIndex.from_frame(raw.iloc[non_test][PAIR].drop_duplicates())
    return dict(pairs_with_eight_test_recipes=len(eligible),
                pairs_absent_from_train_and_validation=len(eligible.difference(seen)))


def author_source(cohort):
    raw = pd.read_csv(verify(AUTHOR, AUTHOR_SHA), usecols=PAIR+RECIPE+["Product SMILES","Yield"])
    np.testing.assert_array_equal(cohort.row_id.astype(int), np.arange(len(raw)))
    np.testing.assert_array_equal(cohort.yield_pct, raw.Yield)
    left = raw[PAIR+RECIPE].apply(lambda row: ".".join(str(value).strip() for value in row), axis=1)
    np.testing.assert_array_equal(cohort.left, left)
    np.testing.assert_array_equal(cohort.right, raw["Product SMILES"].str.strip())
    return raw


def main():
    release_digest = read(RELEASE/"COMPLETE.json")["exported_outputs_sha256"]["yield/cells.csv"]
    records = pd.read_csv(verify(RELEASE/"yield/cells.csv", release_digest), low_memory=False)
    plan_digest = read(ROOT/"plan_seal.json")["sha256"]
    plan = read(verify(ROOT/"plan.json", plan_digest))
    all_scores, all_pairs = [], []
    for dataset, n_rows in (("ncs_bh",27497),("uspto_yield",143535)):
        table = Path(plan["tables"][dataset]["path"])
        cohort = pd.read_csv(verify(table, plan["pinned_files"][str(table)]), dtype={"row_id":str})
        assert len(cohort) == n_rows and cohort.row_id.is_unique
        splits = [entry for entry in plan["splits"] if entry["dataset"] == dataset]
        expected = ({f"i{i}_f{f}" for i in range(10) for f in range(4)}
                    if dataset == "ncs_bh" else {f"pubdup_f{f}" for f in range(5)})
        assert len(splits) == len(expected) and {entry["split"] for entry in splits} == expected
        raw = author_source(cohort) if dataset == "ncs_bh" else None
        for entry in splits:
            path = Path(entry["path"])
            with np.load(verify(path, plan["pinned_files"][str(path)]), allow_pickle=False) as saved:
                selectors = {key:saved[key] for key in PARTS}
            np.testing.assert_array_equal(np.sort(np.concatenate(list(selectors.values()))), np.arange(n_rows))
            split = entry["split"]
            matches = records.loc[(records.dataset == dataset) & (records.split == split)
                                  & records.method.isin(METHODS)]
            assert set(matches.method) == set(METHODS) and len(matches) == len(METHODS)
            all_scores.extend(audit_predictions(row, selectors, cohort) for row in matches.itertuples())
            if raw is not None:
                all_pairs.append(dict(dataset=dataset, split=split, **pair_coverage(raw, selectors)))
    assert len(all_scores) == 180 and len(all_pairs) == 40
    path = HERE / "publication_metadata/preflight_v1"
    path.mkdir(parents=True, exist_ok=True)
    tables = {"predictions_summary.csv":all_scores, "condition_coverage.csv":all_pairs}
    for name, rows in tables.items():
        with (path/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(phase="prediction_and_condition_coverage_preflight",
                   request_commit="d531791d985f574d6703d5740af6d563a665e8f8",
                   verified_prediction_sets=len(all_scores), author_sha256=AUTHOR_SHA,
                   release_table_sha256=release_digest, plan_sha256=plan_digest,
                   pairs_with_eight_test_recipes=sum(r["pairs_with_eight_test_recipes"] for r in all_pairs),
                   pairs_absent_from_train_and_validation=sum(r["pairs_absent_from_train_and_validation"] for r in all_pairs),
                   max_absolute_r2_difference=max(abs(r["r2_difference"]) for r in all_scores),
                   versions={p:version(p) for p in ("numpy","pandas")},
                   outputs_sha256={name:digest(path/name) for name in tables})
    (path/"PREFLIGHT.json").write_text(json.dumps(summary,indent=2,sort_keys=True) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
