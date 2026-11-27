"""Bind all 43,381 archived author rows and their native ordering."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from common90 import AUTHOR, N, ORIGINAL, RELEASE, ROOT, SOURCE, Pins, read, sha, write


def normalize(value):
    parts = []
    for side in value.split(">"):
        if not side:
            parts.append("")
            continue
        molecule = Chem.MolFromSmiles(side)
        if molecule is None:
            raise ValueError("An author-admitted side refused canonical molecular parsing")
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    if len(parts) != 3:
        raise ValueError("Expected a three-field author reaction")
    return ">".join(parts)


def main():
    ROOT.mkdir(exist_ok=True)
    pins = Pins()
    p = SOURCE/"plan.json"
    plan = read(pins.add(p,read(pins.add(SOURCE/"plan_seal.json"))["sha256"]))
    historical = read(pins.add(SOURCE/"COMPLETE.json"))
    archived = read(pins.add(SOURCE/"NATIVE_COMPLETE.json"))
    assert archived["plan_sha256"] == sha(p) and archived["native_train_valid_rows"]==N
    for f in ("protocol.json","comparison_cells.csv"):
        a = RELEASE/f
        pins.add(a,historical["outputs_sha256"][str(a)])
    protocol = read(RELEASE/"protocol.json")
    assert protocol["comparison"] == plan["comparison"] and protocol["native_method"] == plan["native_method"]
    get_native = lambda f: pins.add(SOURCE/"production"/f,archived["artifacts_sha256"][str(SOURCE/"production"/f)])
    author = pd.read_csv(pins.add(AUTHOR,plan["sources_sha256"][str(AUTHOR)]),dtype={"id":str})
    native_fp = get_native("fingerprints.npz")
    with np.load(native_fp,allow_pickle=False) as f:
        pos,ids = f["source_positions"],f["row_ids"]
    rows = author.iloc[pos]
    assert len(pos)==N and len(set(pos))==N
    np.testing.assert_array_equal(ids,rows.id)
    expected = np.concatenate([np.sort(pos[rows.split.eq(s)]) for s in ("train","valid")])
    np.testing.assert_array_equal(pos,expected)
    f = SOURCE/"matched_synrfp/features.npz"
    feature = read(pins.add(f.with_suffix(".json")))
    with np.load(pins.add(f,feature["sha256"]),allow_pickle=False) as a:
        np.testing.assert_array_equal(a["source_positions"],pos)
        np.testing.assert_array_equal(a["row_ids"],ids)
        assert np.all(a["ok"]) and int(a["n_bits"])==2048
        np.save(ROOT/"synrfp_bits.npy",a["X"])
    with np.load(get_native("partition_full.npz"),allow_pickle=False) as partitions:
        np.savez(ROOT/"cohort.npz",source_positions=pos,row_ids=ids,
                 labels=rows["class"].to_numpy(dtype=int),
                 raw=partitions["raw_labels"],residual=partitions["merged_labels"])
    RDLogger.DisableLog("rdApp.*")
    with ProcessPoolExecutor(max_workers=10) as pool:
        reactions=list(pool.map(normalize,rows.rxn_smiles,chunksize=128))
    with (ROOT/"inputs.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("source_position","row_id","rxn"))
        writer.writerows(zip(pos,ids,reactions,strict=True))
    pins.add(ORIGINAL/"Experiments/35_specialist_references/syncluster/compare.py")
    write(ROOT/"PREPARED.json",dict(n=N, distinct_author_ids=len(set(ids)),
          native_fingerprint_path=str(native_fp), input_sha256=pins.files,
          local_sha256={f:sha(ROOT/f) for f in ("cohort.npz","inputs.csv","synrfp_bits.npy")},
          kmeans=protocol["comparison"],native=protocol["native_method"],upstream=protocol["upstream_commit"]))
    print("Prepared",N,"native author records",flush=True)


if __name__=="__main__":
    main()
