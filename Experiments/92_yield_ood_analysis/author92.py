"""Verify the author's deposited bit vectors against released identity and split metadata."""
from ast import literal_eval
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preflight import AUTHOR, AUTHOR_SHA, PAIR, RECIPE, digest, verify
from rank92 import WORK


NAME = "2026-02-21_df_HTE_BH_QM_10iter_9-3clustertest.pkl"
BYTES = 1405922485
MD5 = "c088ab8b56c78ceb381910a05265791c"
AUTHOR_CODE = "3a44e398d9fed3f08cf396f1153be2960193f581"
CODE_SHA = "404b12949082d38a11ba1b591d4288fef35a0ecac7649b9f83c247745d8725aa"
CLUSTERS = [f"iteration_{i} cluster" for i in range(10)]
ASSIGNMENTS = [f"iteration_{i} train_test" for i in range(10)]
IDENTITY = ["Source", *PAIR, *RECIPE, "Product SMILES", *CLUSTERS]
END_BITS = r"^\[(\d)\s+(\d)\s+(\d)\s+\.\.\.\s+(\d)\s+(\d)\s+(\d)\]$"


def identity(frame):
    fields = frame[IDENTITY].copy()
    for col in IDENTITY:
        if col not in CLUSTERS:
            fields[col] = fields[col].astype("string").str.strip()
    return fields.reset_index(drop=True)


def align(author, released, bits):
    expected, got = identity(released), identity(author)
    if expected.equals(got):
        return np.arange(len(author)), "full_identity_in_release_order"
    # Match by explicit source index if the deposit records the original order.
    positions = author.index.get_indexer(released["Unnamed: 0"]) if author.index.is_unique else []
    if len(positions)==len(author) and (positions>=0).all() and expected.equals(got.iloc[positions].reset_index(drop=True)):
        return positions, "release_index_and_full_identity"
    rows = got.assign(_p=np.arange(len(author)))
    # A duplicate key is safe only if its bit vectors are identical.
    duplicate = rows.duplicated(IDENTITY,keep=False)
    for _, group in rows[duplicate].groupby(IDENTITY,dropna=False):
        a = bits[group["_p"].to_numpy()]
        if not np.all(a == a[0]):
            raise ValueError("Reordered author rows contain ambiguous fingerprint identities")
    for frame in (expected,rows):
        frame["_repeat"] = frame.groupby(IDENTITY,dropna=False).cumcount()
    order = expected.merge(rows,on=IDENTITY+["_repeat"],how="left",validate="one_to_one")
    if not order["_p"].is_unique or order["_p"].isna().any():
        raise ValueError("The full author deposit does not align to the released CSV")
    return order["_p"].to_numpy(dtype=int), "full_identity_join"


def assignment(value):
    if isinstance(value,str):
        value = literal_eval(value)
    return tuple(value)


def prepare(plan):
    source = WORK/NAME
    expected = (WORK/"AUTHOR_PICKLE.sha256").read_text().split()[0]
    verify(source,expected)
    code = verify(WORK/"author_drfp_cluster.py",CODE_SHA)
    if source.stat().st_size != BYTES:
        raise ValueError("Unexpected author deposit size")
    with source.open("rb") as file:
        if file_digest(file,"md5").hexdigest()!=MD5:
            raise ValueError("Author deposit does not match the published checksum")
    released = pd.read_csv(verify(AUTHOR,AUTHOR_SHA),low_memory=False)
    author = pd.read_pickle(source)
    assert len(released)==len(author)==27497
    values = np.stack(author["DRFP"].to_numpy())
    assert values.shape==(27497,2048) and np.all((values==0)|(values==1))
    bits = np.packbits(values.astype(bool),axis=1)
    pos, mode = align(author,released,bits)
    a, bits, values = author.iloc[pos], bits[pos], values[pos]
    np.testing.assert_allclose(a.Yield.to_numpy(),released.Yield.to_numpy(),atol=1e-12,rtol=0)
    np.testing.assert_array_equal(a[CLUSTERS].to_numpy(),released[CLUSTERS].to_numpy())
    visible = released["DRFP"].str.extract(END_BITS)
    assert visible.notna().all().all()
    np.testing.assert_array_equal(np.c_[values[:,:3],values[:,-3:]],visible.to_numpy(int))
    tests = {}
    for i,col in enumerate(ASSIGNMENTS):
        raw = np.array([assignment(x) for x in released[col]])
        if col in a:
            np.testing.assert_array_equal(raw,np.array([assignment(x) for x in a[col]]))
        assert raw.shape==(len(released),4)
        for f in range(4):
            tests[f"i{i}_f{f}"] = np.flatnonzero(raw[:,f]=="test")
    for entry in plan["splits"]:
        if entry["dataset"]!="ncs_bh":
            continue
        path = Path(entry["path"])
        with np.load(verify(path,plan["pinned_files"][str(path)]),allow_pickle=False) as split:
            np.testing.assert_array_equal(np.sort(split["test"]),tests[entry["split"]])
    packed = WORK/"author_drfp_aligned.npy"
    np.save(packed,bits,allow_pickle=False)
    receipt = dict(deposit="https://zenodo.org/records/19636649",bytes=BYTES,md5=MD5,sha256=expected,
        clustering_code_commit=AUTHOR_CODE,clustering_code_sha256=digest(code),
        feature_column="DRFP",rows=len(bits),width=values.shape[1],alignment=mode,
        visible_drfp_end_bits_matched=len(bits),matched_heldout_selectors=len(tests),
        aligned_sha256=digest(packed))
    (WORK/"AUTHOR_ALIGNED.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(receipt,flush=True)
    return bits,receipt["aligned_sha256"]
