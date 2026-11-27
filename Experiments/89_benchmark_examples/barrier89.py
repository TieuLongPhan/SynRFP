"""Replay the ten grouped-fold matches and pick paired error extremes."""
from itertools import groupby
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem

from common89 import BASE, ORIGINAL, TMP, WORKSPACE, canonical_sides, feature_row, load_npz, read
import barrier78 as barrier_tokens

DATASETS = ("gdb7","rgd1")
N_ROWS = {"gdb7":11926,"rgd1":353984}
SELECTED_PRESET = "synrfp-wl-r2-b2048-parity-delta_union"
PERIODIC = re.compile(r":\d+\]")


def replay(y, prediction, rec):
    assert y.shape == prediction.shape and np.isfinite(y).all() and np.isfinite(prediction).all()
    residual = prediction.astype(float)-y
    mse = np.mean(residual**2)
    stats = {"rmse":np.sqrt(mse),"mae":np.abs(residual).mean(),"r2":1-mse/y.var()}
    for key,value in stats.items():
        np.testing.assert_allclose(value,rec[key],atol=2e-6,rtol=1e-7)


def syn_fold(pins, row):
    r = read(pins.use(row.source_result_path,row.source_result_sha256))
    assert r["phase"] == "complete" and row.configuration == SELECTED_PRESET
    assert r["full_retained_weight_prediction_replay"]

    def load(name):
        path = Path(r["attempt_directory"])/name
        return load_npz(pins.use(path,r["artifacts_sha256"][str(path)]))

    split, obs = load("rows.npz"), load("evaluation.npz")
    np.testing.assert_array_equal(split["test"],obs["source_positions"])
    for key in ("mae","rmse","r2"):
        np.testing.assert_allclose(r[key],row[key],atol=1e-12,rtol=1e-12)
    replay(obs["observed"],obs["prediction"],r)
    return split,obs


def gdb7_fold(pins, split, release, table):
    folder = TMP/"barrier_allfps_v1/models/gdb7"/f"{split}_f100/chemprop_v2_default"
    wrapper = read(pins.use(folder/"result.json",release.native_result_sha256))
    path = "chemprop/result.json"
    r = read(pins.use(folder/path,release.chemprop_result_sha256))
    assert wrapper["artifacts"][path] == release.chemprop_result_sha256
    assert r["checkpoint_replay_bit_identical"]
    keys = "test_predictions.npz","splits.json"
    al = [pins.use(folder/"chemprop"/name,r["artifacts_sha256"][name]) for name in keys]
    assert r["artifacts_sha256"][keys[0]] == release.test_predictions_sha256
    assert r["data_sha256"] == wrapper["artifacts"]["data.csv"]
    model_rows = pd.read_csv(pins.use(folder/"data.csv",r["data_sha256"]),keep_default_na=False)
    np.testing.assert_array_equal(table.reaction,model_rows.reaction)
    np.testing.assert_allclose(table.barrier_kcal_mol,model_rows.target,rtol=0,atol=1e-10)
    evals = load_npz(al[0])
    replay(evals["y_true"], evals["y_pred"], r)
    return read(al[1]),evals,model_rows.reaction.to_numpy()


def rgd1_fold(pins, split, release, table):
    name = f"rgd1__ea__{split}"
    work = BASE/"rgd1_dmpnn_fullrow_v1"
    proof = read(pins.use(work/"verified"/f"{name}.json"))
    assert proof["result_sha256"] == release.result_sha256
    assert proof["result_path"] == release.result_path and proof["full_source_rows"] == len(table)
    result_path = pins.use(proof["result_path"],proof["result_sha256"])
    r = read(result_path)
    native = result_path.parent / r["native_result_path"]
    n = read(pins.use(native,r["native_result_sha256"]))
    assert r["full_checkpoint_replay"] and n["checkpoint_replay_bit_identical"]
    job = read(pins.use(r["original_job_path"],r["job_file_sha256"]))
    assert job["split"] == split and job["job_id"] == name
    assert job["reaction_csv_sha256"] == r["reaction_csv_sha256"]
    mapped = pd.read_csv(pins.use(job["reaction_csv"],job["reaction_csv_sha256"]),keep_default_na=False,
                         usecols=[job["row_id_column"],job["reaction_column"],job["target_column"]],
                         dtype={job["row_id_column"]:str})
    np.testing.assert_array_equal(mapped.r_id,table.r_id)
    np.testing.assert_allclose(mapped.ea,table.ea,rtol=0,atol=1e-9)
    values = load_npz(pins.use(native.parent/"test_predictions.npz",n["artifacts_sha256"]["test_predictions.npz"]))
    part = read(pins.use(native.parent/"splits.json",n["artifacts_sha256"]["splits.json"]))
    np.testing.assert_array_equal(values["r_id"],mapped.r_id.to_numpy()[values["row_index"]])
    replay(values["y_true"],values["y_pred"],n)
    return part,values,mapped.reaction.to_numpy()


def exact_training_count(fold_rows, frame, dataset, position):
    frame78, archived, original_ok = barrier_tokens.STATE["frames"][dataset]
    delta_bits = archived[0]

    def vector(pos):
        left, right = frame78.left.iat[pos], frame78.right.iat[pos]
        repaired = not original_ok[pos]
        a,b = [barrier_tokens.side_tokens(side,repaired) for side in (left,right)]
        F = barrier_tokens.STATE["native"]
        d,_,_ = F.token_algebra(a,b)
        np.testing.assert_array_equal(F._packed(a,b,"parity",2048,"delta"),delta_bits[pos])
        return d

    assert frame.left.iat[position] == frame78.left.iat[position] and frame.right.iat[position] == frame78.right.iat[position]
    test = vector(position)
    # Identical exact signed counts must produce the same archived bits. Every
    # training candidate with those bits is then compared as full token counts.
    matching_bits = np.all(delta_bits[fold_rows] == delta_bits[position],axis=1)
    shortlist = fold_rows[matching_bits]
    return int(sum(vector(int(i)) == test for i in shortlist)), int(len(shortlist)), len(test)


def structural_descriptors(text):
    fields = re.split(r"(?<!-)>",text)
    sides = []
    for components in (fields[0],fields[-1]):
        molecules = [Chem.MolFromSmiles(r) for r in filter(None,components.split("."))]
        if not molecules or any(m is None for m in molecules):
            raise ValueError("The selected reaction has an unreadable display component")
        sizes = sorted(len(ring) for m in molecules for ring in m.GetRingInfo().AtomRings())
        elements = sorted({a.GetSymbol() for m in molecules for a in m.GetAtoms() if a.GetAtomicNum()>1})
        sides.append((sum(m.GetNumHeavyAtoms() for m in molecules),sizes,elements))
    return {f"{side}_{what}":value for side,fields in zip(("reactant","product"),sides,strict=True)
            for what,value in zip(("heavy_atoms","ring_sizes","elements"),fields,strict=True)}


def compare(pins, used):
    barrier_tokens.TARGETS = DATASETS
    original_counts = barrier_tokens.prepare(pins)
    grouped_folder = WORKSPACE/"SynRFP-barrier-grouped-export/Experiments/72_completed_references/publication_metadata/20261123/dmpnn/grouped_cyclo_gdb7"
    exp = read(pins.use(grouped_folder/"AUDIT.json"))
    a = pd.read_csv(pins.use(grouped_folder/"cells.csv",exp["outputs_sha256"]["cells.csv"]))
    b = pd.read_csv(pins.use(BASE/"rgd1_dmpnn_fullrow_v1/report/cells.csv"))
    fp = ORIGINAL/"Experiments/65_default_optimal/publication_metadata/default_optimal_v1/barrier/cells.csv"
    release = pd.read_csv(pins.use(fp,exp["fingerprint_release_cells_sha256"]))
    outputs,inventory = [],{}
    for dataset,refs in zip(DATASETS,(a,b),strict=True):
        spec,y = original_counts[dataset]
        assert len(y) == N_ROWS[dataset]
        table = pd.read_csv(pins.use(spec["table"]),keep_default_na=False,dtype={"row_id":str,"r_id":str})
        syn_pred = np.full(len(y),np.nan)
        nn_pred = np.full(len(y),np.nan)
        fold_of = np.full(len(y),-1,dtype=int)
        train = {}
        mapped = None
        for fold in range(5):
            key = f"group_cv_f{fold}"
            r = release.loc[release.dataset.eq(dataset) & release.split.eq(key) & release.method.eq("synrfp_optimal")]
            native = refs.loc[refs.dataset.eq(dataset) & refs.split.eq(key)]
            assert len(r) == len(native)==1
            splits, se = syn_fold(pins,r.iloc[0])
            read_fold = gdb7_fold if dataset == "gdb7" else rgd1_fold
            nn_split,de, reactions = read_fold(pins,key,native.iloc[0],table)
            for source_name,ref_name in (("train","train"),("validation","val"),("test","test")):
                np.testing.assert_array_equal(splits[source_name],nn_split[ref_name])
            yy = splits["test"]
            np.testing.assert_array_equal(yy,de["row_index"])
            np.testing.assert_array_equal(se["row_ids"],table.row_id.to_numpy()[yy])
            if dataset=="gdb7":
                np.testing.assert_array_equal(se["row_ids"],de["r_id"])
            np.testing.assert_allclose(y[yy],se["observed"],rtol=0,atol=1e-10)
            np.testing.assert_array_equal(se["observed"],de["y_true"])
            assert np.all(fold_of[yy]<0)
            train[fold] = splits["train"]
            fold_of[yy] = fold
            syn_pred[yy]=se["prediction"]
            nn_pred[yy]=de["y_pred"]
            if mapped is None:
                mapped = reactions
            else:
                np.testing.assert_array_equal(mapped,reactions)
        assert (fold_of >= 0).all()
        gap = np.abs(syn_pred-y)-np.abs(nn_pred-y)
        assert np.isfinite(gap).all()
        inventory[dataset] = dict(paired_heldout_rows=len(y),grouped_folds=5,
                                  synrfp_larger_error=int(np.sum(gap>0)),
                                  dmpnn_larger_error=int(np.sum(gap<0)),ties=int(np.sum(gap==0)),
                                  published_fp_config=SELECTED_PRESET)
        for how, advantage, quota in (("fp_larger_error",gap,3),("dmpnn_larger_error",-gap,1)):
            indices = np.argsort(-advantage,kind="stable")
            accepted = 0
            for value,tied in groupby(indices,key=lambda i:advantage[i]):
                if value <= 0:
                    break
                candidates = []
                for i in tied:
                    r = table.iloc[int(i)]
                    reaction = f"{r.left}>>{r.right}"
                    identifier = r.get("r_id",r.row_id)
                    row = feature_row(dataset,"",identifier,reaction,i,float(y[i]))
                    candidates.append((row,i))
                for row, i in sorted(candidates,key=lambda z:(z[0]["n_heavy_atoms"],z[0]["r_id"],z[1])):
                    if row["scaffold"] in used:
                        continue
                    accepted += 1
                    used.add(row["scaffold"])
                    fold = int(fold_of[i])
                    exact, shortlist, tokens = exact_training_count(train[fold],table,dataset,int(i))
                    row.update(pair_id=f"B_{dataset}_{how}_{accepted:02d}",direction=how,fold=fold,
                               reference_barrier=float(y[i]),synrfp_prediction=float(syn_pred[i]),
                               dmpnn_prediction=float(nn_pred[i]), synrfp_absolute_error=float(abs(syn_pred[i]-y[i])),
                               dmpnn_absolute_error=float(abs(nn_pred[i]-y[i])),
                               synrfp_minus_dmpnn_abs_error=float(gap[i]),
                               n_signed_wl_tokens=tokens,training_exact_signed_count_neighbours=exact,
                               training_equal_delta_bits=shortlist,
                               n_training_rows=len(train[fold]),target="dE0" if dataset=="gdb7" else "ea",
                               units=spec["units"],**structural_descriptors(row["rxn"]))
                    if PERIODIC.search(mapped[i]) and canonical_sides(mapped[i])==canonical_sides(row["rxn"]):
                        row.update(mapped_rxn=mapped[i],mapping_origin="saved Chemprop reaction input")
                    outputs.append(row)
                    if accepted==quota:
                        break
                if accepted==quota:
                    break
            assert accepted==quota
    return outputs, inventory
